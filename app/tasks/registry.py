# Legacy compatibility wrapper redirecting to TaskManager in app.tasks.manager
from app.tasks.manager import BaseTask, TaskManager, task_manager, task_registry, TaskRegistry

__all__ = [
    "BaseTask",
    "TaskManager",
    "task_manager",
    "task_registry",
    "TaskRegistry"
]
