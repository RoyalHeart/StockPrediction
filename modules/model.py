import datetime
import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import init
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

from modules.cafef import get_cafef_dataset
from modules.fireant import get_fireant_dataset
from modules.ssi import get_ssi_dataset
from modules.stock import StockCode

init()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide log
print("Number of GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
# fix random seed for reproducibility
tf.random.set_seed(7)


class ModelType(Enum):
    CNN = "CNN"
    LSTM = "LSTM"
    LSTM_DNN = "LSTM_DNN"
    CNN_LSTM = "CNN_LSTM"
    CNN_LSTM_DNN = "CNN_LSTM_DNN"


TIME_STEP = 5
TRAIN_TEST_RATIO = 0.8
EPOCH_SIZE = 1500
BATCH_SIZE = 8
MODEL_TYPE = ModelType.LSTM_DNN


class StockPrediction:
    # for LSTM best is time_step 4/1, lstm 3
    # for CNN best is time_step 10/32
    start_date = datetime.datetime(2022, 1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_dir = "StockPredict"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.00001,
        patience=EPOCH_SIZE * 0.01,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    def __init__(
        self,
        stock_code: StockCode = StockCode.FPT,
    ):
        self.stock_code = stock_code.value
        self.epoch_size = EPOCH_SIZE
        self.batch_size = BATCH_SIZE
        self.model_type = MODEL_TYPE
        self.time_step = TIME_STEP

    def train(
        self,
        start_date: datetime.datetime = datetime.datetime(2022, 1, 1),
        model_type: ModelType = ModelType.CNN,
        epoch_size: int = EPOCH_SIZE,
        batch_size: int = BATCH_SIZE,
        time_step: int = TIME_STEP,
        ratio: float = TRAIN_TEST_RATIO,
    ):
        self.start_date = start_date
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.model_type = model_type
        self.time_step = time_step
        self.train_test_ratio = ratio
        dataset = self.get_dataset()
        self.save_dataset_plot(dataset)
        dataset = self.transform(dataset)
        self.split_dataset(self.train_test_ratio)
        self.create_trainXY_testXY()
        self.reshape_input()
        self.model = self.create_model()
        self.save_model()
        self.evaluate_model()
        self.plot_result(dataset)

    def load_model_time_step(self, version: int):
        time_step_file = open(
            f"./{self.stock_dir}/{self.stock_code}/{version}/info.md", "r"
        )
        for _ in range(4):
            time_step_file.readline()
        time_step_str = time_step_file.readline()
        return int(time_step_str[11:])

    def get_latest_version(self) -> int:
        version = 0
        for dir in os.scandir(f"./{self.stock_dir}/{self.stock_code}"):
            if dir.is_dir():
                version = int(dir.name) if int(dir.name) > version else version
        return version

    def load_model(self, version) -> Sequential:
        model = tf.keras.models.load_model(
            f"./{self.stock_dir}/{self.stock_code}/{version}/"
        )
        return model

    def predict(
        self,
        predict_number,
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        version=0,
    ):
        if version == 0:
            version = self.get_latest_version()
        model = self.load_model(version)
        time_step = self.load_model_time_step(version)
        self.time_step = time_step
        dataset = self.transform(
            self.get_dataset(
                limit=self.time_step,
                end_date=end_date,
            )
        )
        last_i_price_close = dataset.copy()
        last_i_price_close = np.reshape(last_i_price_close, (1, self.time_step, 1))
        predicted_prices = []
        for _ in range(predict_number):
            predicted_price = model.predict(last_i_price_close, verbose=0)
            predicted_prices.append(predicted_price)
            last_i_price_close[0] = np.reshape(
                np.append(last_i_price_close[0][1:], [[0]]), (self.time_step, 1)
            )
            last_i_price_close[0][-1] = predicted_price
        last_price = float(self.scaler.inverse_transform([dataset[-1]])[0])
        next_predicted_price = float(
            self.scaler.inverse_transform([predicted_price[0]])[0]
        )
        change_percentage = float(
            (next_predicted_price - last_price) / last_price * 100
        )
        print(
            f"{self.stock_code}:\n  Current {last_price:.4f}\n  Next  {next_predicted_price:.4f}"
        )
        print(
            colored(
                f"  Change {change_percentage:.4f}%",
                "green" if change_percentage >= 0 else "red",
            )
        )
        plt.clf()
        plot_data = np.append(dataset, predicted_prices)
        plot_data = self.scaler.inverse_transform([plot_data])
        fig, ax = plt.subplots()
        ax.plot([x for x in range(len(dataset))], plot_data[0][: len(dataset)], "blue")
        ax.plot(
            [x for x in range(len(dataset) - 1, len(dataset) + predict_number)],
            plot_data[0][len(dataset) - 1 :],
            "green",
        )
        plt.savefig(f"./Predict/{self.stock_code}_predict.png")
        plt.close()
        return next_predicted_price, change_percentage

    def update_model(self):
        version = self.get_latest_version()
        print(version)
        model = self.load_model(version)
        time_step = self.load_model_time_step(version)
        dataset = self.get_dataset()
        dataset = self.transform(dataset)
        x, y = self.create_timestep_dataset(dataset, time_step)
        print(x.shape, y.shape)
        x = np.reshape(x, (-1, time_step, 1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(
            x,
            y,
            epochs=self.epoch_size,
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[self.early_stopping],
        )
        self.model = model
        self.save_model(self.get_latest_version())

    def to_string(self):
        print_list = [
            f"Time step: {self.time_step}",
            f"Ratio: {self.train_test_ratio}",
            f"Model type: {self.model_type}",
            f"Epoch size: {self.epoch_size}",
            f"Batch size: {self.batch_size}",
            f"Date: from {self.start_date} to {self.end_date}",
            f"Model json: {self.model_json}",
        ]
        return "\n".join(print_list)

    def get_dataset(
        self,
        start_date=start_date,
        end_date=(datetime.date.today()),
        limit=2000,
    ) -> np.ndarray[any]:
        self.end_date = end_date
        dataset = get_ssi_dataset(
            stock_code=self.stock_code,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
        self.dataset = dataset
        return dataset

    def save_dataset_plot(self, dataset) -> None:
        plt.clf()
        plt.plot(dataset)
        results_dir = os.path.join(".", f"{self.stock_dir}", f"{self.stock_code}")
        file_name = f"dataset_plot.png"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        figure_path = os.path.join(results_dir, file_name)
        plt.savefig(figure_path)

    def transform(self, dataset):
        dataset = self.scaler.fit_transform(dataset)
        self.dataset = dataset
        return dataset

    # split into train and test sets
    def split_dataset(self, ratio=TRAIN_TEST_RATIO):
        train_size = int(len(self.dataset) * ratio)
        train, test = (
            self.dataset[0:train_size, :],
            self.dataset[train_size : len(self.dataset), :],
        )
        print(len(train), len(test))
        self.train = train
        self.test = test

    # convert an array of values into a dataset matrix
    def create_timestep_dataset(self, dataset, time_step, ahead=0):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1 - ahead):
            a = dataset[i : (i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step + ahead, 0])
        return np.array(dataX), np.array(dataY)

    def create_trainXY_testXY(self):
        trainX, trainY = self.create_timestep_dataset(self.train, self.time_step)
        testX, testY = self.create_timestep_dataset(self.test, self.time_step)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    # reshape input to be [samples, time steps, features]
    def reshape_input(self):
        trainX = np.reshape(self.trainX, (self.trainX.shape[0], self.time_step, 1))
        testX = np.reshape(self.testX, (self.testX.shape[0], self.time_step, 1))
        self.trainX = trainX
        self.testX = testX

    # create and fit the CNN network
    def create_model(self):
        model = Sequential()
        if self.model_type == ModelType.CNN:
            model.add(
                Conv1D(32, (3), activation="relu", input_shape=(self.time_step, 1))
            )
            # model.add(MaxPooling1D((2)))
            model.add(Conv1D(64, (3), activation="relu"))
            # model.add(MaxPooling1D((2)))
            model.add(Conv1D(64, (3), activation="relu"))
            model.add(MaxPooling1D((2)))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(1))
        elif self.model_type == ModelType.LSTM:
            model.add(LSTM(50, input_shape=(self.time_step, 1), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(50, input_shape=(self.time_step, 1), return_sequences=True))
            model.add(LSTM(50, input_shape=(self.time_step, 1)))
            model.add(Dense(1))
        elif self.model_type == ModelType.LSTM_DNN:
            model.add(LSTM(50, input_shape=(self.time_step, 1), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(50, input_shape=(self.time_step, 1), return_sequences=True))
            model.add(LSTM(50, input_shape=(self.time_step, 1)))
            model.add(Dense(16))
            model.add(Dense(8))
            model.add(Dense(1))
        elif self.model_type == ModelType.CNN_LSTM:
            model.add(
                Conv1D(32, (3), activation="relu", input_shape=(self.time_step, 1))
            )
            model.add(LSTM(300, "relu", input_shape=(self.time_step, 1)))
            model.add(Dense(1))
        elif self.model_type == ModelType.CNN_LSTM_DNN:
            model.add(
                Conv1D(32, (5), activation="relu", input_shape=(self.time_step, 1))
            )
            model.add(
                LSTM(32, "relu", input_shape=(self.time_step, 1), return_sequences=True)
            )
            model.add(LSTM(16, "relu", input_shape=(self.time_step, 1)))
            model.add(Dense(16))
            model.add(Dense(8))
            model.add(Dense(1))
        else:
            print("Unknown model type")
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.summary()
        self.model_json = model.to_json()
        model.fit(
            self.trainX,
            self.trainY,
            epochs=self.epoch_size,
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[self.early_stopping],
        )
        return model

    # make predictions
    def evaluate_model(self):
        train_predict = self.model.predict(self.trainX)
        test_predict = self.model.predict(self.testX)
        # invert predictions
        train_predict = self.scaler.inverse_transform(train_predict)
        train_true = self.scaler.inverse_transform([self.trainY])
        test_predict = self.scaler.inverse_transform(test_predict)
        test_true = self.scaler.inverse_transform([self.testY])
        # calculate root mean squared error (RMSE)
        trainScore = np.sqrt(mean_squared_error(train_true[0], train_predict[:, 0]))
        train_score_percentage = np.sqrt(
            np.mean(
                np.square(((train_true[0] - train_predict[:, 0]) / train_true[0])),
                axis=0,
            )
        )
        test_score_percentage = np.sqrt(
            np.mean(
                np.square(((test_true[0] - test_predict[:, 0]) / test_true[0])),
                axis=0,
            )
        )
        train_score_percentage_output = (
            "Train RMSPE: %.4f RMSPE" % train_score_percentage
        )
        test_score_percentage_output = "Test RMSPE: %.4f RMSPE" % test_score_percentage
        train_score_output = "Train Score: %.4f RMSE" % (trainScore)
        testScore = np.sqrt(mean_squared_error(test_true[0], test_predict[:, 0]))
        test_score_output = "Test Score: %.4f RMSE" % (testScore)
        print(train_score_output)
        print(test_score_output)
        print(train_score_percentage_output)
        print(test_score_percentage_output)
        rmse_file = open(
            f"./{self.stock_dir}/{self.stock_code}/{self.version}/info.md", "w"
        )
        rmse_file.write(
            train_score_output
            + "\n"
            + test_score_output
            + "\n"
            + train_score_percentage_output
            + "\n"
            + test_score_percentage_output
            + "\n"
            + self.to_string()
        )
        rmse_file.close()
        self.trainPredict = train_predict
        self.testPredict = test_predict

    def save_model(self, version=0):
        if version == 0:
            for dir in os.scandir(f"./{self.stock_dir}/{self.stock_code}"):
                if dir.is_dir():
                    version = int(dir.name) if int(dir.name) > version else version
            version += 1
        self.version = version
        self.model.save(f"./{self.stock_dir}/{self.stock_code}/{version}")

    def plot_result(self, dataset):
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[
            self.time_step : len(self.trainPredict) + self.time_step, :
        ] = self.trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[
            len(self.trainPredict) + (self.time_step * 2) + 1 : len(self.dataset) - 1, :
        ] = self.testPredict
        # plot baseline and predictions
        plt.clf()
        plt.plot(self.scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        # plt.show()
        plt.savefig(f"./{self.stock_dir}/{self.stock_code}/{self.version}/predict.png")
        plt.close()
