from pydantic import BaseModel

class Topic(BaseModel):
    topic: str

class Topics(BaseModel):
    topics: list[str]
