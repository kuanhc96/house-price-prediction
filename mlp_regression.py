from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch.datasets import load_house_attributes, process_house_attributes
from pyimagesearch.models import create_mlp
import numpy as np
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
max_price = train["price"].max()
train["price"] = train["price"] / max_price
test["price"] = test["price"] / max_price

trainY = train["price"]
testY = test["price"]

(trainX, testX) = process_house_attributes(df, train, test)

model = create_mlp(trainX.shape[1], regress=True)
adam = Adam(learning_rate=1e-3, weight_decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=adam)

model.fit(
    x=trainX,
    y=trainY,
    validation_data=(testX, testY),
    epochs=200,
    batch_size=8
)

predictions = model.predict(testX)
diff = predictions.flatten() - testY
percent_diff = ( diff / testY ) * 100
absolute_percent_diff = abs(percent_diff)

mean = np.mean(absolute_percent_diff)
standard_deviation = np.std(absolute_percent_diff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print(f"average house price: {locale.currency(df['price'].mean(), grouping=True)}, standard deviation: {locale.currency(df['price'].std(), grouping=True)}")
print("mean percent difference: {:.2f}%, standard deviation percent difference: {:.2f}%".format(mean, standard_deviation))