from setuptools import setup
from setuptools import find_packages

setup(
    name="maps",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "pyglet"],
    author="Matteo Bettini",
    description="Multi Agent Particle Simulator",
)
