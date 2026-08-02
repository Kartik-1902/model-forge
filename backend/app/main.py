from contextlib import asynccontextmanager
from fastapi import FastAPI
from .config import get_settings

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Before startup execution")
    yield
    print("After shutdown execution")

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan = lifespan
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/version")
    def version():
        return{
            'app': settings.app_name,
            'version': settings.app_version,
            'environment': settings.environment
        }

    return app

app = create_app()