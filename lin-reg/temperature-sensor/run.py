import pandas as pd
import numpy as np
import math

from linear_reg import *

df = pd.read_csv("./data_sensor_temp.csv")[['Pressure', 'Humidity', 'Temperature']]
df = df.dropna()

X = df[['Pressure', 'Humidity']].to_numpy()
y = df['Temperature'].to_numpy()

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

for alpha in [0.0001, 0.00001, 0.000001]:
    mean_cost, std_cost = cross_val(X, y, alpha=alpha)
    print(f"alpha={alpha} → CV cost: {mean_cost:.4f} ± {std_cost:.4f}")
