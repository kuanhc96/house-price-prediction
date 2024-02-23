# python mlp_regression.py -d ./Houses-dataset/
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os 

def load_house_attributes(input_path):
    columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(input_path, sep=" ", header=None, names=columns)

    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # drop the rows with zipcodes that do not appear at least 25 times in the dataset
    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            # df["zipcode"] == zipcode is kind of like the WHERE clause of a sql query: select * from table where zipcode = XXXX
            # df[df["zipcode"] == zipcode] is kind of like the whole sql query
            # df[df["zipcode"] == zipcode].index returns and Index object representing the row number of the selected row
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)
    
    return df

# input variables `train` and `test` are both pandas `Dataframe`s
def process_house_attributes(df, train, test):
    continuous_variables = ["bedrooms", "bathrooms", "area"]
    min_max_scaler = MinMaxScaler()

    # scale the continuous values between [0, 1]
    train_continuous = min_max_scaler.fit_transform(train[continuous_variables])
    test_continuous = min_max_scaler.transform(test[continuous_variables])

    binarizer = LabelBinarizer().fit(df["zipcode"])
    train_categorical = binarizer.transform(train["zipcode"])
    test_categorical = binarizer.transform(test["zipcode"])

    trainX = np.hstack([ train_continuous, train_categorical ])
    testX = np.hstack([ test_continuous, test_categorical ])

    return (trainX, testX)

def load_house_images(df, input_path):
    # initialize images array
    images = []

    # loop over the houses
    for i in df.index.values:
        base_path = os.path.sep.join([input_path, f"{i + 1}_*"])
        house_paths = sorted(list(glob.glob(base_path)))

        input_images = []
        output_image = np.zeros((64, 64, 3), dtype="uint8")

        # loop over images of a given house
        for house_path in house_paths:
            image = cv2.imread(house_path)
            image = cv2.resize(image, (32, 32))
            input_images.append(image)

        output_image[0:32, 0:32] = input_images[0] 
        output_image[0:32, 32:64] = input_images[1] 
        output_image[32:64, 32:64] = input_images[2] 
        output_image[32:64, 0:32] = input_images[3] 
        
        images.append(output_image)

    return np.array( images )