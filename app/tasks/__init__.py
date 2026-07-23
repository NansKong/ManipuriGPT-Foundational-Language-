from app.tasks.manager import task_manager, BaseTask, TaskManager
from app.tasks.registry import task_registry, TaskRegistry

# Import task definitions so they automatically register with task_manager
import app.tasks.translation
import app.tasks.instruction
import app.tasks.chat
import app.tasks.pretraining

__all__ = [
    "task_manager",
    "TaskManager",
    "task_registry",
    "TaskRegistry",
    "BaseTask"
]
