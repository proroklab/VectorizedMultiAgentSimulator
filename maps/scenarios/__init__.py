#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import importlib
import os.path as osp


def load(name: str):
    if name.startswith("simple"):
        name = "mpe/" + name
    pathname = osp.join(osp.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location("", pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
