from .main import ThetaDataPatchProcessor
import importlib
import pkgutil
from pathlib import Path

path = Path(__file__).parent
for module_info in pkgutil.iter_modules([str(path)]):
    module_name = module_info.name
    if module_name != "main":
        importlib.import_module(f".{module_name}", package=__package__)


__all__ = ["ThetaDataPatchProcessor"]