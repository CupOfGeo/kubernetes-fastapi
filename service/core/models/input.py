from pydantic import BaseModel, Field


class MessageInput(BaseModel):
    pass

class PrompInput(BaseModel):
    text: str = Field(..., title="promp")
    max_tokens: int = Field(..., title="max tokens")
    temp: float = Field(..., title="temperature")



