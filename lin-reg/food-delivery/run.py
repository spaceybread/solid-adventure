import pandas as pd
import numpy as np
import math

from linear_reg import *

df = pd.read_csv("./Food_Delivery_Times.csv")[['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']]
df = df.dropna()

X = df[['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']].to_numpy()
y = df['Delivery_Time_min'].to_numpy()

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

for alpha in [0.001, 0.0001, 0.00001]:
    mean_cost, std_cost = cross_val(X, y, alpha=alpha)
    print(f"alpha={alpha} → CV cost: {mean_cost:.4f} ± {std_cost:.4f}")

