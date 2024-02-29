#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from setuptools import find_packages, setup

setup(
    name="vmas",
    version="1.4.0",
    description="Vectorized Multi-Agent Simulator",
    url="https://github.com/proroklab/VectorizedMultiAgentSimulator",
    license="GPLv3",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["numpy", "torch", "pyglet<=1.5.27", "gym", "six"],
    include_package_data=True,
)
