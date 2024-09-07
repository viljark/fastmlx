"""Main module."""

import argparse
import asyncio
from datetime import datetime
import gc
import hashlib
import json
import os
import time
import mlx
from typing import Any, Dict, List, Generator
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import scan_cache_dir

from .types.chat.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
)
from .types.model import SupportedModels

try:
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template as apply_vlm_chat_template
    from mlx_vlm.utils import load_config

    from .utils import (
        MODEL_REMAPPING,
        MODELS,
        apply_lm_chat_template,
        get_eom_token,
        get_tool_prompt,
        handle_function_calls,
        lm_generate,
        lm_stream_generator,
        load_lm_model,
        load_vlm_model,
        vlm_stream_generator,
    )

    MLX_AVAILABLE = True
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")
    MLX_AVAILABLE = False
import time
from typing import Callable

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.routing import APIRoute


class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
                    # Pretty-print the JSON response
            if response.headers.get('Content-Type') == 'application/json':
                try:
                    print('response', json.dumps(response.body.decode(), indent=4))
                except json.JSONDecodeError:
                    print(f"Response is not valid JSON: {response.text()}")
            else:
                print(f"Response is not JSON: {response.headers.get('Content-Type')}")
            print(f"route response headers: {response.headers}")    
            # print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler
        

app = FastAPI()
router = APIRouter(route_class=TimedRoute)

class ModelProvider:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.last_access_times = {}
        self.generating_count: Dict[str, int] = {}
        self.settings: Dict[str, int] = {
            'keep_alive': 5,
            'max_loaded_models': 1,
        }

    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.models

    async def load_model(self, model_name: str):
        async with self.lock:
            model_id = model_name
            if model_name not in self.models:
                print('loading model:', model_name)
                if len(self.models) >= self.settings['max_loaded_models']:
                    models_to_remove = [loaded_model_name for loaded_model_name in self.models.keys()]
                    for loaded_model_name in models_to_remove:
                        while self.generating_count[loaded_model_name] > 0:
                            print('waiting for model to finish generating', loaded_model_name, self.generating_count[loaded_model_name])
                            await asyncio.sleep(1)
                        print('unloading idle model', loaded_model_name)
                        await self.remove_model(loaded_model_name, no_lock= True)

                models = self.get_cached_and_local_models()

                for model in models:
                    if model["id"] == model_name:
                        model_name = model["local_path"] if model["local_path"].startswith("./models") else model_name
                        model_id = model["id"]
                        break

                config = load_config(model_name)
                model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
                if model_type in MODELS["vlm"]:
                    self.models[model_id] = load_vlm_model(model_name, config)
                else:
                    self.models[model_id] = load_lm_model(model_name, config)
                self.generating_count[model_id] = 0
                # Update last access time for the loaded model

            # Update last access time for the requested model
            self.last_access_times[model_id] = time.time()

            return self.models[model_id]

    async def remove_model(self, model_name: str, no_lock = False) -> bool:
        def remove() -> bool:
            if model_name in self.models:
                del self.models[model_name]
                del self.last_access_times[model_name]
                del self.generating_count[model_name]
                print(f"Unloaded model {model_name}")
                gc.collect()  # Force garbage collection
                mlx.core.metal.clear_cache()
                return True
            return False
        if (no_lock):
            return remove()
        else: 
            async with self.lock:
                return remove()

    async def unload_inactive_models(self):
        while True:
            current_time = time.time()

            models_to_unload = [
                    model_name for model_name, last_access in self.last_access_times.items()
                    if current_time - last_access > (self.settings['keep_alive'] * 60) and self.generating_count.get(model_name, 0) == 0
                ]
            tasks = [self.remove_model(model_name) for model_name in models_to_unload]
            await asyncio.gather(*tasks)
            await asyncio.sleep(30)

    async def get_available_models(self):
        async with self.lock:
            return list(self.models.keys())
        

    def get_local_models(self, models_dir = './models'):
        models = []

        # Get a list of all folders (subdirectories) in the models directory
        model_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

        for folder in model_folders:
            model_path = os.path.join(models_dir, folder)
            safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            total_size = 0
            for file in safetensors_files:
                file_path = os.path.join(model_path, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size

            total_size_gb = total_size / (1024 * 1024 * 1024)
            models.append({
                "id": folder,
                "object": "model",
                "type": "",
                "size": f"{total_size_gb:.2f}GB",
                "nb_files": 0,  # Assuming nb_files is not directly available in the folder
                "last_accessed": "N/A",  # Assuming last_accessed is not directly available in the folder
                "last_modified": "N/A",  # Assuming last_modified is not directly available in the folder
                "local_path": model_path,
            })

        return models

    def get_cached_and_local_models(self) -> List[Dict[str, str]]:
        hf_cache_info = scan_cache_dir()
        models = []
        for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path):
            if any(pattern in repo.repo_id.lower() for pattern in ["MLX", "mlx", "4bit", "8bit"]):
                models.append({
                    "id": repo.repo_id,
                    "object": "model",
                    "type": repo.repo_type,
                    "size": "{:>12}".format(repo.size_on_disk_str),
                    "nb_files": repo.nb_files,
                    "last_accessed": repo.last_accessed_str,
                    "last_modified": repo.last_modified_str,
                    "local_path": str(repo.repo_path)
                })

        local_models = self.get_local_models()

        return models + local_models
    
    def start_generating(self, model_name: str):
        self.generating_count[model_name] = self.generating_count.get(model_name, 0) + 1
        print(f"Generating {model_name} ({self.generating_count[model_name]})")

    def stop_generating(self, model_name: str):
        if model_name in self.generating_count:
            self.generating_count[model_name] = max(0, self.generating_count[model_name] - 1)
            print(f"Stopping generation of {model_name}, {self.generating_count[model_name]}")

    def stream_wrapper(self, model_name: str, generator: Generator) -> Generator:
        try:
            yield from generator
        finally:
            self.stop_generating(model_name)

    def set_keep_alive(self, keep_alive: int = 0):
        print("Setting keep alive to", keep_alive)
        self.settings['keep_alive'] = keep_alive

model_provider = ModelProvider()


# Custom type function
def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")


def calculate_default_workers(workers: int = 2) -> int:
    """Calculate the default number of workers based on environment variable."""
    if num_workers_env := os.getenv("FASTMLX_NUM_WORKERS"):
        try:
            workers = int(num_workers_env)
        except ValueError:
            workers = max(1, int(os.cpu_count() * float(num_workers_env)))
    return workers


# Add CORS middleware
def setup_cors(app: FastAPI, allowed_origins: List[str]):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def startup_event():
    print("Starting unload_inactive_models task")
    asyncio.create_task(model_provider.unload_inactive_models())
def get_keys(dictionary):
    keys = []
    if isinstance(dictionary, list):
        for item in dictionary:
            keys.extend(get_keys(item))
    elif isinstance(dictionary, dict):
        for key in dictionary:
            keys.append({'key', dictionary[key][:10]})
            keys.extend(get_keys(dictionary[key]))
    return keys
print

@router.post("/v1/completions", response_model=ChatCompletionResponse)
async def completions(request: CompletionRequest):
    if not MLX_AVAILABLE:
        raise HTTPException(status_code=500, detail="MLX library not available")

    stream = request.stream
    model_data = await model_provider.load_model(request.model)
    model = model_data["model"]
    config = model_data["config"]
    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])

    stop_words = get_eom_token(request.model)
    if model_type in MODELS["vlm"]:
        # Implementation for VLM model for completions
        pass
    else:
        # Implementation for LM model for completions
        tokenizer = model_data["tokenizer"]
        prompt = request.prompt  # Assuming the prompt is part of the request

        # Generate the completion
        output, token_length_info  = lm_generate(
            model,
            tokenizer,
            prompt,
            request.max_tokens,
            temp=request.temperature,
            stop_words=stop_words,
        )

        if stream:
            generator = lm_stream_generator(
                model,
                request.model,
                tokenizer,
                prompt,
                request.max_tokens,
                request.temperature,
                legacy=True,
                stop_words=stop_words,
                stream_options=request.stream_options,
            )
            return StreamingResponse(
                model_provider.stream_wrapper(request.model, generator),
                media_type="text/event-stream",
            )
        else:
            # Prepare the response
            response = ChatCompletionResponse(
                id=f"cmpl-{os.urandom(16).hex()}",
                object="text_completion",
                created=int(time.time()),
                model=request.model,
                usage=token_length_info,
                choices=[
                    {
                        "text": output,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            )

        model_provider.stop_generating(request.model)

        return response
    
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):

    if not MLX_AVAILABLE:
        raise HTTPException(status_code=500, detail="MLX library not available")

    if "gemma-2" in request.model.lower():
        # Remove system role messages for gemma-2 model
        request.messages = [msg for msg in request.messages if msg.role!= "system"]

    stream = request.stream
    model_data = await model_provider.load_model(request.model)
    model = model_data["model"]
    config = model_data["config"]
    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
    stop_words = get_eom_token(request.model)
    model_provider.start_generating(request.model)

    if model_type in MODELS["vlm"]:
        processor = model_data["processor"]
        image_processor = model_data["image_processor"]

        # Set image from the latest message if found
        # TODO: not working as expected
        image = None
        for message in reversed(request.messages):
            for item in message.content:
                if 'image_url' in item:
                    image = item['image_url']['url']
                    break  # Stop searching once the image URL is found
                else:
                    continue  # Continue to the next message if no image URL is found in this message
            break  # Stop looping through messages once an image URL is found

        chat_messages = []

        for msg in request.messages:
            content = []
            if isinstance(msg.content, list):
                # Handle image_url in content
                for item in msg.content:
                    if item["type"] == "text":
                        content.append(item["text"])
            else:
                content = msg.content
            chat_messages.append({"role": msg.role, "content": content})

        print(chat_messages)

        prompt = ""
        if model.config.model_type != "paligemma":
            prompt = apply_vlm_chat_template(processor, config, chat_messages)
        else:
            prompt = request.messages[-1].content
        if stream:
            generator = vlm_stream_generator(
                    model,
                    request.model,
                    processor,
                    image,
                    prompt,
                    image_processor,
                    request.max_tokens,
                    request.temperature,
                    stop_words=stop_words,
                    stream_options=request.stream_options,
                )

            return StreamingResponse(
                model_provider.stream_wrapper(request.model, generator),
                media_type="text/event-stream",
            )
        else:
            # Generate the response
            output = vlm_generate(
                model,
                processor,
                image,
                prompt,
                image_processor,
                max_tokens=request.max_tokens,
                temp=request.temperature,
                verbose=False,
            )

    else:
        # Add function calling information to the prompt
        if request.tools and "firefunction-v2" not in request.model:
            # Handle system prompt
            if request.messages and request.messages[0].role == "system":
                pass
            else:
                # Generate system prompt based on model and tools
                prompt, user_role = get_tool_prompt(
                    request.model,
                    [tool.model_dump() for tool in request.tools],
                    request.messages[-1].content,
                )

                if user_role:
                    request.messages[-1].content = prompt
                else:
                    # Insert the system prompt at the beginning of the messages
                    request.messages.insert(
                        0, ChatMessage(role="system", content=prompt)
                    )

        tokenizer = model_data["tokenizer"]

        chat_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        prompt = apply_lm_chat_template(tokenizer, chat_messages, request)

        if stream:
                generator = lm_stream_generator(
                    model,
                    request.model,
                    tokenizer,
                    prompt,
                    request.max_tokens,
                    request.temperature,
                    stop_words=stop_words,
                    stream_options=request.stream_options,
                )
                return StreamingResponse(
                    model_provider.stream_wrapper(request.model, generator),
                    media_type="text/event-stream",
                )
        else:
            output, token_length_info  = lm_generate(
                model,
                tokenizer,
                prompt,
                request.max_tokens,
                temp=request.temperature,
                stop_words=stop_words,
            )

    # Parse the output to check for function calls
    result = handle_function_calls(output, request, token_length_info)
    model_provider.stop_generating(request.model)

    return result

@router.head("/")
async def head():
    return 'fast-mlx is running'

@router.get("/")
async def index():

    return 'fast-mlx is running'

def convert_size_to_bytes(size_string):
    # Remove any whitespace and convert to uppercase
    size_string = size_string.strip().upper()
    
    # Extract the numeric part and the unit
    number = float(size_string[:-1])
    unit = size_string[-1]
    
    if unit == 'G':
        return int(number * 1e6)  # 1 billion
    else:
        raise ValueError("Unsupported unit. Expected 'B' for billion.")

# shoul be ollama compliant, but enchanted is still not picking it up
@router.get("/api/tags")
async def get_tags():
    models = model_provider.get_cached_and_local_models()
    
    models = []
    for model in models:
        # Generate a pseudo-digest using the model id
        digest = hashlib.sha256(model['id'].encode()).hexdigest()
        
        # Parse the last_modified date
        try:
            modified_at = datetime.strptime(model['last_modified'], "%Y-%m-%d %H:%M:%S").isoformat()
        except ValueError:
            modified_at = datetime.now().isoformat()  # Use current time if parsing fails
        
        # Extract parameter size and quantization level from the model id
        parameter_size = "Unknown"
        quantization_level = "Unknown"
        if "-" in model['id']:
            parts = model['id'].split("-")
            for part in parts:
                if part.endswith("B"):
                    parameter_size = part
                elif part.startswith("Q"):
                    quantization_level = part
        
        models.append({
            "name": model['id'],
            "model": model['id'],
            "modified_at": modified_at,
            "size": convert_size_to_bytes(model["size"]),
            "digest": digest,
            "details": {
                "format": "gguf",  # Assuming GGUF format for all models
                "family": "llama" if "llama" in model['id'].lower() else "unknown",
                "families": None,
                "parameter_size": parameter_size,
                "quantization_level": quantization_level
            }
        })
    
    return {"models": models}

@app.get("/v1/supported_models", response_model=SupportedModels)
async def get_supported_models():
    """
    Get a list of supported model types for VLM and LM.
    """
    return JSONResponse(content=MODELS)


@app.get("/v1/models")
async def list_models():
    models = model_provider.get_cached_and_local_models()
    model_ids = [m['id'] + ' ' + m['size'] for m in models]
    print('available models\n', json.dumps(model_ids, indent=2))
    return {
        "object": "list",
        "data": models
    }

@app.post("/v1/models")
async def add_model(model_name: str):
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@app.delete("/v1/models")
async def remove_model(model_name: str):
    model_name = unquote(model_name).strip('"')
    removed = await model_provider.remove_model(model_name)
    if removed:
        return Response(status_code=204)  # 204 No Content - successful deletion
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


def run():
    parser = argparse.ArgumentParser(description="FastMLX API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=False,
        help="Enable auto-reload of the server. Only works when 'workers' is set to None.",
    )

    parser.add_argument(
        "--workers",
        type=int_or_float,
        default=calculate_default_workers(),
        help="""Number of workers. Overrides the `FASTMLX_NUM_WORKERS` env variable.
        Can be either an int or a float.
        If an int, it will be the number of workers to use.
        If a float, number of workers will be this fraction of the  number of CPU cores available, with a minimum of 1.
        Defaults to the `FASTMLX_NUM_WORKERS` env variable if set and to 2 if not.
        To use all available CPU cores, set it to 1.0.

        Examples:
        --workers 1 (will use 1 worker)
        --workers 1.0 (will use all available CPU cores)
        --workers 0.5 (will use half the number of CPU cores available)
        --workers 0.0 (will use 1 worker)""",
    )

    
    parser.add_argument(
        "--keep-alive",
        type=int,
        default=0,
        help="Time in minutes to keep models loaded after last access. 0 means no unloading."
    )

    args = parser.parse_args()
    if isinstance(args.workers, float):
        args.workers = max(1, int(os.cpu_count() * args.workers))

    # does not work, second thread not picking up the updated value, hardcoded to 5 in ModelProvider
    model_provider.set_keep_alive(args.keep_alive)
    setup_cors(app, args.allowed_origins)

    import uvicorn

    uvicorn.run(
        "fastmlx:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        loop="asyncio",
        lifespan="on"
    )


if __name__ == "__main__":
    run()


app.include_router(router)