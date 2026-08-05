from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from app.auth.keys import KeyStore
from app.auth.schemas import APIKeyInfo, APIKeyResponse, CreateKeyRequest
from app.dependencies import get_db_session

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/keys", response_model=APIKeyResponse)
async def create_key(request: CreateKeyRequest, db: AsyncSession = Depends(get_db_session)) -> APIKeyResponse:
    store = KeyStore(db)
    raw_key, api_key = await store.create_key(request.name)
    return APIKeyResponse(name=api_key.name, key=raw_key, created_at=api_key.created_at)


@router.get("/keys", response_model=list[APIKeyInfo])
async def list_keys(db: AsyncSession = Depends(get_db_session)) -> list[APIKeyInfo]:
    store = KeyStore(db)
    keys = await store.list_keys()
    return [
        APIKeyInfo(
            name=k.name,
            key_prefix=k.key_prefix,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.delete("/keys/{key_prefix}")
async def revoke_key(key_prefix: str, db: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
    store = KeyStore(db)
    revoked = await store.revoke_key(key_prefix)
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")
    return {"message": "Key revoked successfully"}
