#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import importlib
import os
import os.path as osp
from pathlib import Path


def load(name: str):
    pathname = None
    for dirpath, _, filenames in os.walk(osp.dirname(__file__)):
        if pathname is None:
            for filename in filenames:
                if name == filename or name == str(Path(dirpath) / Path(filename)):
                    pathname = os.path.join(dirpath, filename)
                    break
    assert pathname is not None, f"{name} scenario not found."

    spec = importlib.util.spec_from_file_location("", pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
