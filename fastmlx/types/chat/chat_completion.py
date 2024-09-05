from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]  # content can be either text or a list of text/image_url objects

class ImageURL(BaseModel):
    url: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    prompt: str = Field(default="")
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=16000)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.2)
    tools: Optional[List[Function]] = Field(default=None)
    tool_choice: Optional[str] = Field(default=None)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

class CompletionRequest(BaseModel):
    model: str
    prompt: str = Field(default="")
    max_tokens: int = Field(default=16000)
    temperature: float = Field(default=0.3, gt=0, le=1)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = Field(default=False)
    logprobs: Union[int, None] = Field(default=None)
    echo: bool = Field(default=False)
    stop: Union[str, List[str], None] = Field(default=None)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    best_of: int = Field(default=1, ge=1, le=10)
    user: str = Field(default="")
