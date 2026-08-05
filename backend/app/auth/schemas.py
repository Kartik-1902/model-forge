from datetime import datetime

from pydantic import BaseModel, Field


class CreateKeyRequest(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=100, description="Human-readable label for this API key"
    )


class APIKeyResponse(BaseModel):
    name: str
    key: str
    created_at: datetime


class APIKeyInfo(BaseModel):
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None
