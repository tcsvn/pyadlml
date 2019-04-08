from loadUniPenData import loadUnipenData
from plotting import plotUniPenData, plotVoronoid
from normalize import normalize_example
import numpy as np

# Load Unipen Data and labels
data, labels = loadUnipenData('pendigits-orig.tra')

# normalize data leaving PenUp and PenDown
data, tdata = normalize_example(data)
data2 = np.array(data)