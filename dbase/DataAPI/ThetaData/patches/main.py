from typing import Any, Callable, Dict, List, Tuple
import inspect
from pathlib import Path
import yaml
import importlib
from trade.helpers.Logging import setup_logger
logger = setup_logger("dbase.DataAPI.ThetaData.PatchProcessor", "WARNING")

class ThetaDataPatchProcessor:
    """
    Class responsible for applying patches to ThetaData API functions to handle known data issues.
    """

    PATCH_FUNCTIONS: Dict[str, List[Callable]] = {}

    @classmethod
    def register_patch(cls, func_name: str, patch_func: Callable):
        """
        Register a patch function for a specific ThetaData API function.
        Parameters    ----------
        func_name : str
            The name of the ThetaData API function to patch.
        patch_func : Callable
            The patch function to apply to the specified API function.

        Note:
        - Return value has to be the first parameter of the patch function.
        - Return value parameter has to be named "result" for the patch processor to recognize it as the output of the patch function.
        - Patch functions should return result regardless of whether they apply a patch or not. If the patch does not apply, they should return the original result without modification.
        - Function must allow for arbitrary additional parameters using *args and **kwargs to ensure compatibility with the API functions they are patching.
        """
        logger.info(f"Registering patch for {func_name} with function {patch_func.__name__}")
        if func_name not in cls.PATCH_FUNCTIONS:
            cls.PATCH_FUNCTIONS[func_name] = []
        certified, message = cls.certify_function_signature(patch_func)
        if not certified:
            raise ValueError(f"Patch function {patch_func.__name__} does not have a valid signature: {message}")
        if patch_func not in cls.PATCH_FUNCTIONS[func_name]:
            cls.PATCH_FUNCTIONS[func_name].append(patch_func)

    @classmethod
    def apply_patches(cls, func_name: str, result: Any, *args, **kwargs) -> Any:
        """
        Apply all registered patches for a specific ThetaData API function.
        Parameters    ----------
        func_name : str
            The name of the ThetaData API function to apply patches to.
        result : Any
            The original result from the ThetaData API function before applying patches.
        *args
            Positional arguments to pass to the patch functions.
        **kwargs
            Keyword arguments to pass to the patch functions.
        Returns    -------
        The result of applying the patches, if any, or None if no patches are registered."""

        if func_name in cls.PATCH_FUNCTIONS:
            for patch_func in cls.PATCH_FUNCTIONS[func_name]:
                result = patch_func(result, *args, **kwargs)
                if result is not None:
                    return result
        else:
            logger.info(f"No patches registered for {func_name}")
        return result
    
    @classmethod
    def certify_function_signature(cls, patch_func: Callable) -> Tuple[bool, str]:
        """
        Certify the signature of a patch function to ensure it is compatible with the ThetaData API functions.
        Parameters    ----------
        patch_func : Callable
            The patch function to inspect.
        Returns    -------
        bool
            True if the patch function has a valid signature, False otherwise."""
        sig = inspect.signature(patch_func)
        params = sig.parameters

        # Result parameter exists
        if "result" not in params:
            return False, "Missing 'result' parameter"
        
        # Result parameter is the first parameter
        if list(params.keys())[0] != "result":
            return False, "'result' parameter must be the first parameter"
        
        # Patch function allows for arbitrary additional parameters
        if not any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()) or not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return False, "Patch function must allow for arbitrary additional parameters using *args and **kwargs"
        return True, "Valid signature"
    
    @classmethod
    def setup(cls):
        """
        Setup method to initialize the patch processor and register all patches from setup.yaml.
        """
        loc = Path(__file__).parent / "setup.yaml"
        if not loc.exists():
            logger.warning(f"Setup file {loc} does not exist. No patches will be registered.")
            return
        with open(loc, "r") as f:
            config = yaml.safe_load(f)
            for patch in config.get("patches", []):
                func_name = patch.get("func_name")
                patch_func_name = patch.get("patch_func")
                mod_name = patch.get("function_module")

                if func_name and patch_func_name and mod_name:
                    try:
                        mod = importlib.import_module(mod_name)
                        patch_func = getattr(mod, patch_func_name)
                        cls.register_patch(func_name, patch_func)
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Failed to register patch {patch_func_name} for {func_name}: {e}")
