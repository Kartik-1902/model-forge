from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.keys import KeyStore
from app.config import get_settings
from app.dependencies import get_db_session

api_key_header = APIKeyHeader(name=get_settings().api_key_header, auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header), db: AsyncSession = Depends(get_db_session)
) -> str:
    """
    Dependency to verify API keys for standard endpoints.
    Returns the name of the key if valid.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
        )

    store = KeyStore(db)
    key_meta = await store.validate_key(api_key)

    if key_meta is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    return key_meta.name


async def verify_admin_key(
    api_key: str | None = Security(api_key_header), db: AsyncSession = Depends(get_db_session)
) -> str:
    """
    Dependency to verify admin access.
    Allows access if the key matches BOOTSTRAP_ADMIN_KEY,
    otherwise falls back to standard API key verification.
    """
    settings = get_settings()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
        )

    # Check bootstrap key first
    if settings.bootstrap_admin_key and api_key == settings.bootstrap_admin_key:
        return "bootstrap_admin"

    # Fall back to standard key validation
    store = KeyStore(db)
    key_meta = await store.validate_key(api_key)

    if key_meta is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    return key_meta.name
