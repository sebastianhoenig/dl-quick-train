"""Bypass-import the ``BatchTopKSAE`` class without triggering
``dictionary_learning/__init__.py`` (which pulls in nnsight → transformers →
torchvision, a chain that can be broken in a given environment).

Only used by analysis scripts that evaluate existing SAE checkpoints.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types


def _load_batch_top_k_sae_class():
    if "dictionary_learning.trainers.batch_top_k" in sys.modules:
        return sys.modules["dictionary_learning.trainers.batch_top_k"].BatchTopKSAE

    try:
        import dictionary_learning  # noqa: F401
        # Happy path: environment is fine.
        from dictionary_learning.trainers.batch_top_k import BatchTopKSAE  # type: ignore
        return BatchTopKSAE
    except Exception:
        pass

    # Fallback: rebuild the package tree by hand, skipping __init__.py side
    # effects that require nnsight.
    try:
        import dictionary_learning as _existing
        pkg_dir = os.path.dirname(_existing.__file__)
    except Exception:
        import importlib.machinery as _m
        spec = _m.PathFinder.find_spec("dictionary_learning")
        if spec is None or spec.submodule_search_locations is None:
            raise RuntimeError("dictionary_learning package not installed")
        pkg_dir = spec.submodule_search_locations[0]

    pkg = types.ModuleType("dictionary_learning")
    pkg.__path__ = [pkg_dir]
    sys.modules["dictionary_learning"] = pkg
    trainers_pkg = types.ModuleType("dictionary_learning.trainers")
    trainers_pkg.__path__ = [os.path.join(pkg_dir, "trainers")]
    sys.modules["dictionary_learning.trainers"] = trainers_pkg

    def _load(name: str, rel_path: str):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(pkg_dir, rel_path)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("dictionary_learning.dictionary", "dictionary.py")
    _load("dictionary_learning.config", "config.py")
    _load("dictionary_learning.trainers.trainer", "trainers/trainer.py")
    bt = _load(
        "dictionary_learning.trainers.batch_top_k", "trainers/batch_top_k.py"
    )
    return bt.BatchTopKSAE


BatchTopKSAE = _load_batch_top_k_sae_class()


__all__ = ["BatchTopKSAE"]
