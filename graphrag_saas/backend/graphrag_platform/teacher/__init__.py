from .base import Teacher
from .local import LocalTeacher
from .ollama import OllamaTeacher
from .schema import Citation, TeacherOutputV1

__all__ = [
    "Citation",
    "Teacher",
    "TeacherOutputV1",
    "LocalTeacher",
    "OllamaTeacher",
]
