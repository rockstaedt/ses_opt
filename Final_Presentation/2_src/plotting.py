###############################################################################
### Plotting
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
import json
from pathlib import Path
import os

from model_options import output

###############################################################################
### Matplotlib Settings
###############################################################################

settings = {
    'text.usetex': True,
    'font.weight' : 'normal',
    'font.size'   : 14
}
plt.rcParams.update(**settings)

# resolution for plots
dpi = 300

current_path = Path.cwd()

# path to csv files
results_path = os.path.join(current_path.parent, '3_results')

# saving path
saving_path = os.path.join(current_path.parent, '4_plots')
