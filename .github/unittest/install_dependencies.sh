#
# Copyright (c) ProrokLab.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


python -m pip install --upgrade pip

pip install -e ".[gymnasium]"

python -m pip install flake8 pytest pytest-cov tqdm matplotlib
python -m pip install cvxpylayers # Navigation heuristic
