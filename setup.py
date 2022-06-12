#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

from setuptools import setup, find_packages

setup(
    name="maps",
    version="0.0.1",
    description="Multi Agent Particle Simulator",
    url="https://github.com/proroklab/MultiAgentParticleSimulator",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["torch", "numpy", "pyglet", "gym"],
    include_package_data=True,
    zip_safe=False,
)
