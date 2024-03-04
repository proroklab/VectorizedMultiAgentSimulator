#
# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.
#

sudo apt-get update
sudo apt-get install python3-opengl xvfb

python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov tqdm

pip install -e .
