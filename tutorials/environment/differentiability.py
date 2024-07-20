#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
Data download example
=====================

This example shows one way of dealing with large data files required for your
examples.

The ``download_data`` function first checks if the data has already been
downloaded, looking in either the data directory saved the configuration file
(by default ``~/.sg_template``) or the default data directory. If the data has
not already been downloaded, it downloads the data from the url and saves the
data directory to the configuration file. This allows you to use the data
again in a different example without downloading it again.

Note that examples in the gallery are ordered according to their filenames, thus
the number after 'plot_' dictates the order the example appears in the gallery.
"""


######################################################################
# Define Hyperparameters
# ----------------------
#
# We set the hyperparameters for our tutorial.
# Depending on the resources
# available, one may choose to execute the policy and the simulator on GPU or on another
# device.
# You can tune some of these values to adjust the computational requirements.
#

import vmas

env = vmas.make_env("dispersion", num_envs=3)
env.get_random_actions()

######################################################################
# Define dwdwd
# ----------------------
#
# We set the hyperparameters for our tutorial.
# Depending on the resources
# available, one may choose to execute the policy and the simulator on GPU or on another
# device.
# You can tune some of these values to adjust the computational requirements.
#

env = vmas.make_env("dispersion", num_envs=3)
env.get_random_actions()
