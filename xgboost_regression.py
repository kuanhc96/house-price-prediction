from sklearn.model_selection import train_test_split
import xgboost as xgb
from pyimagesearch.datasets import load_house_attributes, process_house_attributes
import numpy as np
import pandas as pd
import argparse
import locale
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of house images")
args = vars(ap.parse_args())

input_path = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = load_house_attributes(input_path)

(train, test) = train_test_split(df, test_size=0.25, random_state=42)

# scale prices column
max_price = df["price"].max()
train["price"] = train["price"] / max_price
test["price"] = test["price"] / max_price

trainY = train["price"]
testY = test["price"]

(trainX, testX) = process_house_attributes(df, train, test)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    reg_lambda=2,
    gamma=0,
    max_depth=8
)

xgb_model.fit(trainX, trainY, verbose=True)

# feature_importance = pd.DataFrame(xgb_model.feature_importances_.reshape(1, -1), columns=train.columns)
# print(feature_importance)

predictions = xgb_model.predict(testX)
diff = predictions.flatten() - testY
percent_diff = ( diff / testY ) * 100
absolute_percent_diff = abs(percent_diff)

mean = np.mean(absolute_percent_diff)
standard_deviation = np.std(absolute_percent_diff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print(f"average house price: {locale.currency(df['price'].mean(), grouping=True)}, standard deviation: {locale.currency(df['price'].std(), grouping=True)}")
print("mean percent difference: {:.2f}%, standard deviation percent difference: {:.2f}%".format(mean, standard_deviation))