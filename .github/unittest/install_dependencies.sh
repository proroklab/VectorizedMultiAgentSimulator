#
# Copyright (c) 2024-2025.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.
#


python -m pip install --upgrade pip

pip install -e ".[gymnasium]"

python -m pip install flake8 pytest pytest-cov tqdm matplotlib
python -m pip install cvxpylayers # Navigation heuristic
