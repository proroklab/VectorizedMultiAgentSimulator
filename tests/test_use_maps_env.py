#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import unittest

from examples.use_maps_env import use_maps_env


class TestUseMapsEnv(unittest.TestCase):
    def test_use_maps_env(self):
        use_maps_env()
