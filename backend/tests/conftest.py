import pytest_asyncio
from httpx import ASGITransport, AsyncClient
# pyrefly: ignore [missing-import]
from app.main import create_app
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
# pyrefly: ignore [missing-import]
from app.db.models import Base
# pyrefly: ignore [missing-import]
from app.dependencies import get_db_session

# Test database using aiosqlite in-memory
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    # pyrefly: ignore [missing-import]
    from app.tasks.registry import init_registry
    init_registry("app.tasks")
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db_session():
    async with test_session_maker() as session:
        yield session

@pytest_asyncio.fixture
async def test_api_key(db_session):
    # pyrefly: ignore [missing-import]
    from app.auth.keys import KeyStore
    store = KeyStore(db_session)
    raw_key, api_key = await store.create_key("test_key")
    return raw_key, api_key

@pytest_asyncio.fixture
async def client(test_api_key):
    app = create_app()
    
    # Override get_db_session dependency
    async def override_get_db_session():
        async with test_session_maker() as session:
            yield session
    app.dependency_overrides[get_db_session] = override_get_db_session
    
    raw_key, _ = test_api_key
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        headers={"X-API-Key": raw_key}
    ) as ac:
        yield ac
