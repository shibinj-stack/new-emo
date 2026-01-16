from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(50,1)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(4, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model