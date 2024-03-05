Installation
============


Install from PyPi
-----------------

You can install `VMAS <https://github.com/proroklab/VectorizedMultiAgentSimulator>`__ from PyPi.

.. code-block:: console

   pip install vmas

Install from source
-------------------

If you want to install the current main version (more up to date than latest release), you can do:

.. code-block:: console

   git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git
   cd VectorizedMultiAgentSimulator
   pip install -e .


Install optional requirements
-----------------------------

By default, vmas has only the core requirements.
Here are some optional packages you may want to install.

Training
^^^^^^^^

You may want to install one of the following training libraries

.. code-block:: console

   pip install benchmarl
   pip install torchrl
   pip install "ray[rllib]"==2.1.0 # We support versions "ray[rllib]<=2.2,>=1.13"

Logging
^^^^^^^

You may want to install the following rendering and logging tools

.. code-block:: console

   pip install wandb
   pip install opencv-python moviepy matplotlib
