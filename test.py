import numpy as np
import main_informer
from exp.exp_informer import Exp_Informer
import matplotlib.pyplot as plt
import pandas as pd

cpm_data = pd.read_csv("data/CPM/cpm.csv")
cpm_data['recordValue'] = cpm_data['recordValue'].apply(lambda x: x + 200)
cpm_data.to_csv("data/CPM/cpm.csv", index=False)
