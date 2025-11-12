#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
Baby 2
======

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

import vmas

env = vmas.make_env("dispersion", num_envs=3)
print(env.reset())
env.reset()
