import hashlib
import secrets
from datetime import UTC

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import APIKey


class KeyStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_key(self, name: str) -> tuple[str, APIKey]:
        raw_key = f"mf_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:11]

        api_key = APIKey(name=name, key_prefix=key_prefix, key_hash=key_hash, is_active=True)
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        return raw_key, api_key

    async def validate_key(self, raw_key: str) -> APIKey | None:
        key_hash = self._hash_key(raw_key)
        stmt = select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True)
        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()

        if api_key:
            from datetime import datetime

            api_key.last_used_at = datetime.now(UTC)
            await self.session.commit()
            await self.session.refresh(api_key)
            return api_key
        return None

    async def revoke_key(self, key_prefix: str) -> bool:
        stmt = select(APIKey).where(APIKey.key_prefix == key_prefix, APIKey.is_active == True)
        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()
        if api_key:
            api_key.is_active = False
            await self.session.commit()
            return True
        return False

    async def list_keys(self) -> list[APIKey]:
        stmt = select(APIKey).order_by(APIKey.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()
