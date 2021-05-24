import os
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from model import pamk

df = pd.read_excel(os.path.join('data', 'data.xlsx'))
data = df.iloc[:, np.r_[1, 4:10]]

standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(data.iloc[:, 1:].values)

out = pamk(X, method='spectral_pam')

print('test end')
