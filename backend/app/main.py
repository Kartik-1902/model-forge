import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from .config import get_settings
from .tasks.registry import init_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

    from app.api.routes.admin import router as admin_router
    from app.api.routes.tasks import router as tasks_router
    from app.auth.dependencies import verify_admin_key, verify_api_key

    app.include_router(admin_router, dependencies=[Depends(verify_admin_key)])
    app.include_router(tasks_router, dependencies=[Depends(verify_api_key)])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }

    return app


app = create_app()
