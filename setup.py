#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import pathlib

from setuptools import find_packages, setup


def get_version():
    """Gets the vmas version."""
    path = CWD / "vmas" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="vmas",
    version=get_version(),
    description="Vectorized Multi-Agent Simulator",
    url="https://github.com/proroklab/VectorizedMultiAgentSimulator",
    license="GPLv3",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["numpy", "torch", "pyglet<=1.5.27", "gym", "six"],
    extras_require={
        "gymnasium": ["gymnasium", "shimmy"],
        "rllib": ["ray[rllib]<=2.2"],
        "render": ["opencv-python", "moviepy", "matplotlib", "opencv-python"],
        "test": ["pytest", "pytest-instafail", "pyyaml", "tqdm"],
    },
    include_package_data=True,
)
