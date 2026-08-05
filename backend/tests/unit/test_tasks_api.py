import pytest

# pyrefly: ignore [missing-import]


@pytest.mark.asyncio
async def test_list_tasks(client):
    resp = await client.get("/api/v1/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert "tasks" in data
    tasks = data["tasks"]
    assert any(t["task_name"] == "tabular_classification" for t in tasks)


@pytest.mark.asyncio
async def test_get_task(client):
    resp = await client.get("/api/v1/tasks/tabular_classification")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_name"] == "tabular_classification"
    assert "input_schema" in data
    assert "output_schema" in data
    assert "evaluation_metrics" in data


@pytest.mark.asyncio
async def test_get_task_not_found(client):
    resp = await client.get("/api/v1/tasks/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_task_models(client):
    resp = await client.get("/api/v1/tasks/tabular_classification/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "random_forest" in data["models"]


@pytest.mark.asyncio
async def test_list_task_models_not_found(client):
    resp = await client.get("/api/v1/tasks/nonexistent/models")
    assert resp.status_code == 404
