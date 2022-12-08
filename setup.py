#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from setuptools import setup, find_packages

setup(
    name="vmas",
    version="1.2.0",
    description="Vectorized Multi-Agent Simulator",
    url="https://github.com/proroklab/VectorizedMultiAgentSimulator",
    license="GPLv3",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pyglet<=1.5.27",
        "gym<=0.23.1",
    ],
)
