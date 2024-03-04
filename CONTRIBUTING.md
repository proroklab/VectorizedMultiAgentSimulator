# Contributing to VMAS
We want to make contributing to this project as easy and transparent as
possible.

## Installing the library

To contribute, it is suggested to install the library (or your fork of it) from source:

```bash
git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git
cd VectorizedMultiAgentSimulator
python setup.py develop
```

## Formatting your code

Before your PR is ready, you'll probably want your code to be checked. This can be done easily by installing
```
pip install pre-commit
```
and running
```
pre-commit run --all-files
```
from within the vmas cloned directory.

You can also install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`
). You can disable the check by appending `-n` to your commit command: `git commit -m <commit message> -n`

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite and the documentation pass.
5. Make sure your code lints.

When submitting a PR, we encourage you to link it to the related issue (if any) and add some tags to it.


## License
By contributing to vmas, you agree that your contributions will be licensed
under the license of the project
