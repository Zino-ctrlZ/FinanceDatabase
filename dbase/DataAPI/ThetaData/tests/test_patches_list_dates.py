import numpy as np
import importlib.util
import sys
import types
from pathlib import Path


def _load_patch_function():
    """Load patch function without importing the full ThetaData package tree."""
    patch_file = Path(__file__).resolve().parents[1] / "patches" / "p1.py"
    module_name = "dbase.DataAPI.ThetaData.patches.p1"
    main_module_name = "dbase.DataAPI.ThetaData.patches.main"

    if main_module_name not in sys.modules:
        fake_main = types.ModuleType(main_module_name)

        class _NoopPatchProcessor:
            @classmethod
            def register_patch(cls, *args, **kwargs):
                return None

        fake_main.ThetaDataPatchProcessor = _NoopPatchProcessor
        sys.modules[main_module_name] = fake_main

    spec = importlib.util.spec_from_file_location(module_name, patch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load patch module from {patch_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module._remove_incorrect_date_for_option


_remove_incorrect_date_for_option = _load_patch_function()


def test_list_dates_patch_removes_aapl_split_artifact_date():
    result = np.array(
        [
            "2020-08-25",
            "2020-08-31",
            "2020-09-01",
            "2020-09-02",
            "2020-09-03",
        ],
        dtype=object,
    )

    patched = _remove_incorrect_date_for_option(
        result=result,
        symbol="AAPL",
        exp="2020-11-20",
        right="C",
        strike=120.0,
    )

    assert "2020-08-25" not in set(patched.tolist())
    assert patched.tolist() == ["2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03"]


def test_list_dates_patch_does_not_remove_when_discontinuity_pattern_absent():
    result = np.array(
        [
            "2020-08-25",
            "2020-08-26",
            "2020-08-27",
            "2020-08-31",
        ],
        dtype=object,
    )

    patched = _remove_incorrect_date_for_option(
        result=result,
        symbol="AAPL",
        exp="2020-11-20",
        right="C",
        strike=120.0,
    )

    assert patched.tolist() == result.tolist()


def test_list_dates_patch_is_symbol_specific_to_aapl():
    result = np.array(
        [
            "2020-08-25",
            "2020-08-31",
            "2020-09-01",
        ],
        dtype=object,
    )

    patched = _remove_incorrect_date_for_option(
        result=result,
        symbol="MSFT",
        exp="2020-11-20",
        right="C",
        strike=120.0,
    )

    assert patched.tolist() == result.tolist()


def run_regular_tests() -> None:
    test_list_dates_patch_removes_aapl_split_artifact_date()
    test_list_dates_patch_does_not_remove_when_discontinuity_pattern_absent()
    test_list_dates_patch_is_symbol_specific_to_aapl()
    print("ALL_REGULAR_TESTS_PASSED")


if __name__ == "__main__":
    run_regular_tests()
