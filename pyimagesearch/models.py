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

# does the same thing as create_sequential_cnn, just different notations style
# However, this notation is required for creating CNNs that have combined inputs
def create_functional_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    input_shape = (height, width, depth)
    channel_dimension = -1
    inputs = Input(shape=input_shape)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs

        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=channel_dimension)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=channel_dimension)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model

def create_sequential_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    input_shape = (height, width, depth)
    channel_dimension = -1
    model = Sequential()

    # CONV -> RELU -> BN -> POOL
    # loop over filters:
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        if i == 0:
            model.add(Conv2D(f, (3, 3), padding="same", input_shape=input_shape))
        else:
            model.add(Conv2D(f, (3, 3), padding="same"))

        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dimension))
    model.add(Dropout(0.5))

    model.add(Dense(4))
    model.add(Activation("relu"))

    if regress:
        model.add(Dense(1, activation="linear"))

    return model
