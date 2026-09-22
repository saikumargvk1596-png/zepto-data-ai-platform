from pydantic import BaseModel,Field
from typing import List
class AskRequest(BaseModel): query:str=Field(min_length=1)
class AskResponse(BaseModel):
    answer:str
    sources:List[str]=[]
    confidence:float=Field(ge=0,le=1)
