"""Run the model without CHAP, using the default options.

The last 3 months of each location are held out as future data.
"""

import pandas as pd

from predict import predict
from train import train

df = pd.read_csv("input/training_data.csv").sort_values(["location", "time_period"])
future = df.groupby("location").tail(3)
historic = df.drop(future.index)
historic.to_csv("output/historic.csv", index=False)
future.drop(columns="disease_cases").to_csv("output/future.csv", index=False)

train("output/historic.csv", "output/model.pkl")
predict("output/model.pkl", "output/historic.csv", "output/future.csv", "output/predictions.csv")
print("Done! See output/predictions.csv")
