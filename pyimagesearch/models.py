from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input 
from tensorflow.keras.models import Model

def create_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    # if regress, add a regression node; otherwise, it is a classification problem, and the previous layer is good as is
    if regress:
        model.add(Dense(1, activation="linear"))

    return model