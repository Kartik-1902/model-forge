"""
Task & Model Plugin Registry — Auto-Discovery

Discovers TaskDefinition and ModelImplementation subclasses by scanning
the tasks/ package tree at startup.  The registry owns the mapping:

    task_name  →  TaskDefinition (single shared instance)
    task_name  →  {model_name → ModelImplementation class}

Models are stored as *classes*, not instances.  Instantiation (with the
shared TaskDefinition injected via constructor) happens at training /
inference time.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

from app.tasks.base import ModelImplementation, TaskDefinition

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Discovers and stores task definitions and model implementations.

    Usage::

        registry = TaskRegistry()
        registry.discover("app.tasks")
        registry.get_task("tabular_classification")
        registry.get_model("tabular_classification", "random_forest")
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskDefinition] = {}
        self._models: dict[str, dict[str, type[ModelImplementation]]] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, base_package: str) -> None:
        """Scan subpackages of *base_package* for task and model plugins.

        For each subpackage (e.g. ``app.tasks.tabular_classification``):

        1. Import its ``task`` module and find the ``TaskDefinition`` subclass.
        2. Instantiate the ``TaskDefinition`` once (immutable, shared object).
        3. Scan the ``models`` sub-package for ``ModelImplementation`` subclasses.
        4. Register both into internal dictionaries.
        """
        base_module = importlib.import_module(base_package)

        # base_module.__path__ lets pkgutil iterate *direct* sub-packages
        for importer, subpkg_name, is_pkg in pkgutil.iter_modules(
            base_module.__path__, prefix=f"{base_package}."
        ):
            if not is_pkg:
                # Skip non-package modules (base.py, registry.py, __init__.py)
                continue

            self._discover_task(subpkg_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_task(self, task_package: str) -> None:
        """Import ``<task_package>.task`` and register the TaskDefinition."""
        task_module_name = f"{task_package}.task"
        try:
            task_module = importlib.import_module(task_module_name)
        except ModuleNotFoundError:
            logger.warning("Skipping %s — no 'task' module found", task_package)
            return

        task_cls = self._find_subclass(task_module, TaskDefinition)
        if task_cls is None:
            logger.warning(
                "Skipping %s — no TaskDefinition subclass in %s",
                task_package,
                task_module_name,
            )
            return

        # Instantiate once — immutable, shared across all models of this task
        task_instance = task_cls()
        task_name = task_instance.task_name

        if task_name in self._tasks:
            logger.warning(
                "Duplicate task name '%s' — overwriting previous registration",
                task_name,
            )

        self._tasks[task_name] = task_instance
        self._models.setdefault(task_name, {})

        logger.debug("Registered task: %s (%s)", task_name, task_cls.__name__)

        # Now discover models inside <task_package>.models
        self._discover_models(task_package, task_name)

    def _discover_models(self, task_package: str, task_name: str) -> None:
        """Import modules inside ``<task_package>.models`` and register
        every ``ModelImplementation`` subclass found."""
        models_package_name = f"{task_package}.models"
        try:
            models_package = importlib.import_module(models_package_name)
        except ModuleNotFoundError:
            logger.debug(
                "No 'models' sub-package in %s — task has no model implementations yet",
                task_package,
            )
            return

        for importer, module_name, is_pkg in pkgutil.iter_modules(
            models_package.__path__, prefix=f"{models_package_name}."
        ):
            if is_pkg:
                continue  # We only look at modules, not nested packages

            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, ModelImplementation)
                    and obj is not ModelImplementation
                    and not inspect.isabstract(obj)
                ):
                    # We store the *class*, not an instance.
                    # The registry (or caller) injects the shared TaskDefinition
                    # when it needs to construct one.
                    model_name = self._resolve_model_name(obj, task_name)

                    if model_name in self._models[task_name]:
                        logger.warning(
                            "Duplicate model '%s' for task '%s' — overwriting",
                            model_name,
                            task_name,
                        )

                    self._models[task_name][model_name] = obj
                    logger.debug(
                        "  Registered model: %s/%s (%s)",
                        task_name,
                        model_name,
                        obj.__name__,
                    )

    @staticmethod
    def _find_subclass(module: Any, base: type) -> type | None:
        """Return the first concrete subclass of *base* found in *module*,
        or ``None``."""
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base) and obj is not base and not inspect.isabstract(obj):
                return obj
        return None

    @staticmethod
    def _resolve_model_name(model_cls: type[ModelImplementation], task_name: str) -> str:
        """Instantiate the model temporarily to read ``model_name``.

        This is the only reliable way to get the name because it is an
        abstract *property* — we cannot read it from the class object
        without an instance.  The instance is thrown away immediately.
        """
        # We need a TaskDefinition to construct the model.  We use a
        # lightweight approach: instantiate with a minimal task just to
        # read the property.  But we already have the real task stored
        # in _tasks, so the caller should pass it.
        #
        # For simplicity (and because model_name should not depend on the
        # task), we peek at the class attribute directly if it exists.
        if hasattr(model_cls, "_model_name"):
            return str(getattr(model_cls, "_model_name"))

        # Fallback: derive from the class name
        # e.g. RandomForestModel → random_forest_model
        # Prefer explicit over implicit, but this covers the common case.
        cls_name = model_cls.__name__
        # Convert CamelCase to snake_case
        import re

        name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", cls_name)
        name = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", name)
        return name.lower()

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_task(self, task_name: str) -> TaskDefinition:
        """Return the shared TaskDefinition for *task_name*.

        Raises ``KeyError`` if no such task is registered.
        """
        try:
            return self._tasks[task_name]
        except KeyError:
            raise KeyError(
                f"Task '{task_name}' is not registered. Available tasks: {list(self._tasks)}"
            ) from None

    def get_model(self, task_name: str, model_name: str) -> type[ModelImplementation]:
        """Return the ModelImplementation *class* for *task_name*/*model_name*.

        Raises ``KeyError`` if the task or model is not registered.
        """
        self.get_task(task_name)  # validate task exists (raises KeyError)
        try:
            return self._models[task_name][model_name]
        except KeyError:
            raise KeyError(
                f"Model '{model_name}' is not registered for task '{task_name}'. "
                f"Available models: {list(self._models.get(task_name, {}))}"
            ) from None

    def list_tasks(self) -> list[str]:
        """Return all registered task names."""
        return list(self._tasks)

    def list_models(self, task_name: str) -> list[str]:
        """Return model names registered for *task_name*.

        Raises ``KeyError`` if no such task is registered.
        """
        self.get_task(task_name)  # validate
        return list(self._models[task_name])

    def get_task_info(self, task_name: str) -> dict[str, Any]:
        """Return a JSON-serialisable dict of task metadata.

        Includes name, input/output schema (JSON Schema), evaluation
        metrics, default thresholds, and registered model names.
        """
        task = self.get_task(task_name)
        return {
            "task_name": task.task_name,
            "input_schema": task.input_schema.model_json_schema(),
            "output_schema": task.output_schema.model_json_schema(),
            "evaluation_metrics": task.evaluation_metrics,
            "default_thresholds": task.get_default_thresholds(),
            "models": self.list_models(task_name),
        }


# ======================================================================
# Module-level singleton
# ======================================================================

_global_registry: TaskRegistry | None = None


def get_registry() -> TaskRegistry:
    """Return the initialised global ``TaskRegistry``.

    Raises ``RuntimeError`` if ``init_registry()`` has not been called.
    """
    if _global_registry is None:
        raise RuntimeError("TaskRegistry not initialised. Call init_registry() first.")
    return _global_registry


def init_registry(base_package: str = "app.tasks") -> TaskRegistry:
    """Create, populate, and store the global ``TaskRegistry``."""
    global _global_registry
    _global_registry = TaskRegistry()
    _global_registry.discover(base_package)
    return _global_registry
