import argparse

import mlflow.tensorflow
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

from mlflow_callback import MlFlowCallback

def data_preparation(df):
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .8))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size')
    parser.add_argument('--epochs')
    parser.add_argument('--company')

    args = parser.parse_args()
    ds = pd.read_csv("../data/data.csv")
    df = ds[ds['company_name'] ==args.company]

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    x_train, y_train=data_preparation(df)

    with mlflow.start_run():

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        # Train the model
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,callbacks=[MlFlowCallback()])