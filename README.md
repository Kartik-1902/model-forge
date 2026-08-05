# ModelForge

ModelForge is a lightweight, modular, production-inspired AI platform for training, deploying, and monitoring ML models. It provides a robust plugin architecture where every model encapsulates both its training and inference logic within a standard `ModelImplementation` contract.

## Prerequisites

- **Python**: `>= 3.12`
- **Package Manager**: [uv](https://astral.sh/uv) (Recommended for lightning-fast, deterministic dependency resolution)

## Quick Start (Local Development)

### 1. Setup Environment
Clone the repository and install all dependencies deterministically:
```bash
uv sync
```

### 2. Configure Environment Variables
Copy the template configuration file:
*(On Windows Powershell)*
```powershell
Copy-Item .env.example .env
```
*(On Unix)*
```bash
cp .env.example .env
```

Ensure the `BOOTSTRAP_ADMIN_KEY` in your `.env` is set to something secure if deploying outside a local environment.

### 3. Start the Server
Start the FastAPI server:
```bash
cd backend
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Generate an API Key (Bootstrapping)
ModelForge is secure by default. To interact with the API, you must first generate an API key using the `BOOTSTRAP_ADMIN_KEY` defined in your `.env` file.

Open a new terminal and run:
```bash
curl -X POST http://localhost:8000/api/v1/admin/keys \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change_me_in_production" \
  -d '{"name":"my-first-key"}'
```
*Note: Replace `change_me_in_production` with the actual value of `BOOTSTRAP_ADMIN_KEY` from your `.env` file.*

The API will return your new API key. **Save it immediately**, as it will not be shown again.

### 5. Verify Access
You can now use your newly generated key to access secured endpoints, such as listing available tasks:
```bash
curl http://localhost:8000/api/v1/tasks \
  -H "X-API-Key: <your_new_api_key>"
```

## Useful Developer Commands

Because ModelForge uses `uv`, you can run development tasks easily without a Makefile. Run these from the project root:

- **Run Unit Tests**: `uv run pytest backend/tests/unit/ -v`
- **Lint Code**: `uv run ruff check .`
- **Format Code**: `uv run ruff format .`
- **Type Checking**: `uv run mypy backend/app/`

## Architecture

ModelForge uses a task-centric plugin architecture:
- **`TaskDefinition`**: Defines input/output schemas and evaluation metrics (e.g., `tabular_classification`).
- **`ModelImplementation`**: Implements training, prediction, and evaluation for a specific framework (e.g., `random_forest`).

The platform automatically discovers plugins at startup by scanning the `backend/app/tasks/` directory.