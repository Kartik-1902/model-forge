from fastapi import APIRouter, HTTPException
from typing import Any

from app.tasks.registry import get_registry

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.get("")
async def list_tasks() -> dict[str, list[dict[str, Any]]]:
    registry = get_registry()
    return {"tasks": [registry.get_task_info(t) for t in registry.list_tasks()]}


@router.get("/{task_name}")
async def get_task(task_name: str) -> dict[str, Any]:
    registry = get_registry()
    try:
        return registry.get_task_info(task_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")


@router.get("/{task_name}/models")
async def list_task_models(task_name: str) -> dict[str, list[str]]:
    registry = get_registry()
    try:
        models = registry.list_models(task_name)
        return {"models": models}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
