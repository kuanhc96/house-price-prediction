from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate
from sklearn.model_selection import train_test_split
from pyimagesearch.datasets import load_house_attributes, process_house_attributes, load_house_images
from pyimagesearch.models import create_mlp, create_functional_cnn
import numpy as np
import argparse
import locale
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset of house images")
args = vars(ap.parse_args())

print("[INFO] load house attributes")
input_path = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = load_house_attributes(input_path)

print("[INFO] load house images")
images = load_house_images(df, args["dataset"])
images = images / 255.0

(Xtrain_attr, Xtest_attr, Xtrain_img, Xtest_img) = train_test_split(df, images, test_size=0.25, random_state=42)

max_price = Xtrain_attr["price"].max()
Ytrain = Xtrain_attr["price"]/max_price
Ytest = Xtest_attr["price"]/max_price

(Xtrain_attr, Xtest_attr) = process_house_attributes(df, Xtrain_attr, Xtest_attr)

cnn_model = create_functional_cnn(64, 64, 3, regress=False)
mlp_model = create_mlp(Xtrain_attr.shape[1], regress=False)
combined = concatenate([ cnn_model.output, mlp_model.output ])
# the input to the combined Fully Connected layer is 8. It makes sense for the next layer to be 4
z = Dense(4, activation="relu")(combined)
z = Dense(1, activation="linear")(z)
combined_model = Model(inputs=[cnn_model.inputs, mlp_model.inputs], outputs=z)

adam = Adam(learning_rate=1e-3, weight_decay=1e-3/200)
combined_model.compile(loss="mean_absolute_percentage_error", optimizer=adam)

combined_model.fit([ Xtrain_img, Xtrain_attr ], Ytrain, validation_data=([ Xtest_img, Xtest_attr ], Ytest), epochs=200, batch_size=8)

predictions = combined_model.predict([ Xtest_img, Xtest_attr ])
diff = predictions.flatten() - Ytest
percent_diff = ( diff / Ytest ) * 100
absolute_percent_diff = abs(percent_diff)

mean = np.mean(absolute_percent_diff)
standard_deviation = np.std(absolute_percent_diff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print(f"average house price: {locale.currency(df['price'].mean(), grouping=True)}, standard deviation: {locale.currency(df['price'].std(), grouping=True)}")
print("mean percent difference: {:.2f}%, standard deviation percent difference: {:.2f}%".format(mean, standard_deviation))