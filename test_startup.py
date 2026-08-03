import asyncio
from app.main import app
from asgi_lifespan import LifespanManager

async def test_lifespan():
    async with LifespanManager(app):
        print("Lifespan started successfully")

if __name__ == "__main__":
    asyncio.run(test_lifespan())
