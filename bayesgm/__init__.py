from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "1.0.2"

if TYPE_CHECKING:
    from . import datasets, models, utils
    from .models.bgm_iv import (
        BGM_IV,
        BGM_IV_Image,
        BGM_IV_Vector,
    )

_SYMBOL_TO_MODULE = {
    "BGM_IV": "bayesgm.models.bgm_iv",
    "BGM_IV_Image": "bayesgm.models.bgm_iv",
    "BGM_IV_Vector": "bayesgm.models.bgm_iv",
}

_MODULE_ATTRIBUTES = {
    "models": "bayesgm.models",
    "datasets": "bayesgm.datasets",
    "utils": "bayesgm.utils",
}

__all__ = [
    "BGM_IV",
    "BGM_IV_Image",
    "BGM_IV_Vector",
]


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(_SYMBOL_TO_MODULE[name])
        return getattr(module, name)
    if name in _MODULE_ATTRIBUTES:
        return import_module(_MODULE_ATTRIBUTES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_MODULE_ATTRIBUTES))
