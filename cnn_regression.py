from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
import numpy as np
import argparse
import locale
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of house images")
args = vars(ap.parse_args())

print("[INFO] load house attributes")
input_path = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(input_path)

print("[INFO] load house images")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0

(Xtrain_attr, Xtest_attr, Xtrain_img, Xtest_img) = train_test_split(df, images, test_size=0.25, random_state=42)

max_price = Xtrain_attr["price"].max()
Ytrain = Xtrain_attr["price"]/max_price
Ytest = Xtest_attr["price"]/max_price

model = models.create_cnn(64, 64, 3, regress=True)
adam = Adam(learning_rate=1e-3, weight_decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=adam)

model.fit(Xtrain_img, Ytrain, validation_data=(Xtest_img, Ytest), epochs=200, batch_size=8)

predictions = model.predict(Xtest_img)
diff = predictions.flatten() - Ytest
percent_diff = ( diff / Ytest ) * 100
absolute_percent_diff = abs(percent_diff)

mean = np.mean(absolute_percent_diff)
standard_deviation = np.std(absolute_percent_diff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print(f"average house price: {locale.currency(df['price'].mean(), grouping=True)}, standard deviation: {locale.currency(df['price'].std(), grouping=True)}")
print("mean percent difference: {:.2f}%, standard deviation percent difference: {:.2f}%".format(mean, standard_deviation))