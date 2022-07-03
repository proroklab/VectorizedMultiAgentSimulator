#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import unittest

from examples.use_vmas_env import use_vmas_env


class TestUseVmasEnv(unittest.TestCase):
    def test_use_vmas_env(self):
        use_vmas_env()
