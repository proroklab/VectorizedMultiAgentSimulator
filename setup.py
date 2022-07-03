#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from setuptools import setup, find_packages

setup(
    name="vmas",
    version="0.0.1",
    description="Vectorized Multi-Agent Simulator",
    url="https://github.com/proroklab/VectorizedMultiAgentSimulator",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["torch", "numpy", "pyglet", "gym"],
    include_package_data=True,
    zip_safe=False,
)
