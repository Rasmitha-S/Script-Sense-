import os
import sys
import importlib.util

# Re-export all SQLAlchemy models from root backend/models.py
models_py_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.py")
spec = importlib.util.spec_from_file_location("root_models", models_py_path)
if spec and spec.loader:
    _root_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_root_models)
    
    Base = _root_models.Base
    User = _root_models.User
    Student = _root_models.Student
    AnswerSheet = _root_models.AnswerSheet
    ModelAnswer = _root_models.ModelAnswer
    Evaluation = _root_models.Evaluation
    FinalResult = _root_models.FinalResult

    for _k, _v in _root_models.__dict__.items():
        if not _k.startswith("__"):
            globals()[_k] = _v

__all__ = [
    "Base",
    "User",
    "Student",
    "AnswerSheet",
    "ModelAnswer",
    "Evaluation",
    "FinalResult",
]
