from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping  # For the machine learning model
from keras.layers import Dense, Activation, Dropout, LSTM  # For the machine learning model
from keras.models import Sequential  # For the machine learning model
from keras.utils import plot_model  # For the machine learning model


def music_model(X, Y, shape_organiser, number_of_epochs=256):
    # Build a stacked LSTM network
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(shape_organiser, 3)))
    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    plot_model(model, to_file="Model.png", show_shapes=True)

    checkpoint_maker = ModelCheckpoint(monitor="loss", filepath="Checkpoint.hdf5",
                                       verbose=1, save_best_only=True)  # Save best points
    reduce_lr = ReduceLROnPlateau(monitor="loss", patience=25, verbose=1,
                                  factor=0.9)  # Reduce learning rate if there is no improvement.
    stop_early = EarlyStopping(monitor="loss", patience=100, verbose=1)

    model.compile(loss='mse', optimizer='adam')

    model.fit(X, Y, batch_size=len(X), epochs=number_of_epochs, callbacks=[checkpoint_maker, reduce_lr, stop_early])

    return model
