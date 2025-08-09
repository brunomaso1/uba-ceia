from pydantic import BaseModel


class UploadJGWDataResponse(BaseModel):
    id: int
    jgw_file_name: str