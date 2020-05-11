"""Import commonly used libraries"""

import numpy as np
import pandas as pd
import collections

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42 # editable text in matplotlib
mpl.rcParams['svg.fonttype'] = 'none'

import matplotlib.ticker as mtick
PercentFormat = mtick.FuncFormatter(lambda y, _: '{:.1%}'.format(y))
ScalarFormat = mtick.ScalarFormatter()


# seaborn
import seaborn as sns
sns.set_style('ticks', rc={'axes.grid':True})
sns.set_context('talk')


