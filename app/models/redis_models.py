
from typing import Optional
from pydantic import BaseModel

class JobEnqueueResponse(BaseModel):
    """
    Response returned when a background job is successfully enqueued.
    """

    job_id: str


class JobStatusResponse(BaseModel):
    """
    Response returned when polling the status of a background job.
    """

    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
