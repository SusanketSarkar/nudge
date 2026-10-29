import shutil
import pytest

from nudge.experiments.config import config_from_dict
from nudge.models.tiny import build_tiny_backend


@pytest.fixture(scope="session")
def backend():
    return build_tiny_backend(seed=0)


@pytest.fixture()
def small_cfg(tmp_path):
    return config_from_dict({
        "experiment": {"name": "t", "seed": 1, "run_dir": str(tmp_path)},
        "backend": {"kind": "tiny"},
        "search": {"layers": [2, 3], "neurons_per_layer": 3, "doses": [-1.0, 1.0]},
        "generation": {"max_new_tokens": 8},
        "controls": {"per_layer_dose": 4, "n_prompts": 5},
        "prompts": {"suite": "fiction_scenes", "n_screening": 5, "n_validation": 5, "n_test": 6},
        "selection": {"shortlist": 3, "finalists": 2, "min_valid_prompts": 3, "control_percentile": 0.0,
                      "require_control": False},
        "criteria": {"min_prompts": 5, "min_control_percentile": 0.0},
    })
