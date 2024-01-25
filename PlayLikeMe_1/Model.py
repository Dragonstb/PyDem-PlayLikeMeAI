import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Concatenate, Dense, Dropout


def setupNewModel(printModel: bool) -> tf.keras.Model:
    """
    Creates a new model

    printModel:
    Call the method summary() for the model.

    return:
    A new, untrained model.
    """
    # inputs:
    # suit of pocket card:      4 (one hot) per card
    # value of pcket card:     13 (one hot) per card
    # stack size:               1 (numeric)
    # current bet:              1 (numeric)
    # seats ahead of me:        1 (numeric) 0 is me
    # position:                 1 (numeric) 0 is the dealer
    # is active:                1 (one hot) 1 if and only if player has status active
    # slot used:                1 (one hot) 1 if the player slot is open and 0 if it is free
    # a total of 40 inputs per player slot, and 10 player slots available
    #
    # pot size:                 1 (numeric)
    # current bet:              1 (numeric)
    # small blind:              1 (numeric)
    # big blind:                1 (numeric)
    # suit of community card:   4 (one hot) per card
    # value of community card: 13 (one hot) per card
    # sums up to a total of 72 inputs

    playerInput = Input(shape=(10, 40))
    flattedPlayerInput = Flatten()(playerInput)
    communityInput = Input(shape=(70, 1))
    flattedCommunityInput = Flatten()(communityInput)
    joined = Concatenate(axis=1)([flattedPlayerInput, flattedCommunityInput])

    dense1 = Dense(512, activation='relu')(joined)
    drop1 = Dropout(0.25)(dense1)
    dense2 = Dense(256, activation='relu')(drop1)
    dense2 = Dense(128, activation='relu')(dense2)
    drop2 = Dropout(0.25)(dense2)
    dense3 = Dense(64, activation='relu')(drop2)
    dense4 = Dense(32, activation='relu')(dense3)
    # remind that a negative results represents folding, so negative values must be a possible
    output = Dense(1, activation='linear')(dense4)

    model = tf.keras.Model(
        inputs=[playerInput, communityInput], outputs=output)

    if printModel:
        model.summary()

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    return model
