import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .config import get_settings
from .tasks.registry import init_registry

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    registry = init_registry()
    tasks = registry.list_tasks()
    logger.info("Discovered %d task(s): %s", len(tasks), tasks)
    for task_name in tasks:
        models = registry.list_models(task_name)
        logger.info("  Task '%s': %d model(s): %s", task_name, len(models), models)
    yield
    # ---- Shutdown ----
    logger.info("Shutting down Model Forge")

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/version")
    def version():
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment
        }

    return app

app = create_app()