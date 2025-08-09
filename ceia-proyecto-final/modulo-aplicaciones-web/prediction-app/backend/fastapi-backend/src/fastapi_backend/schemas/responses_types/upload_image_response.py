from pydantic import BaseModel


class UploadImageResponse(BaseModel):
    id: int
    name: str
