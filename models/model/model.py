"""Titanic Survival Project Example
This file demonstrates how we can develop and train our model by using the
`passenger_features` we've developed earlier. Every ML model project
should have a definition file like this one.
"""
from typing import Any
from sklearn.model_selection import train_test_split
from layer import Featureset, Train
from tensorflow import keras
from sklearn import metrics
import custom_keras_loss


def train_model(train: Train, pf: Featureset("house_features")) -> Any:

    df = pf.to_pandas()

    X = df.drop(["SalePrice", "Id"], axis=1)
    y = df["SalePrice"]

    test_size = 0.2
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)
    train.register_input(X_train)
    train.register_output(y_train)

    model = keras.Sequential()
    model.add(keras.layers.Dense(5,activation='relu',input_shape=(2,)))
    model.add(keras.layers.Dense(5,activation='relu'))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.07), loss = custom_keras_loss.custom_mae)

    epochs = 10
    train.log_parameter("Epochs", epochs)
    model.fit(X_train, y_train, epochs = epochs, validation_data = (X_test,y_test))
    predictions = model.predict(X_test)
    mean_absolute_error = metrics.mean_absolute_error(predictions, y_test)
    train.log_metric("Mean Absolute Error",mean_absolute_error)

    return model