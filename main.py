from modules.model import ModelType, StockCode, StockPrediction


def main():
    vnm = StockPrediction(
        StockCode.VNM,
    )
    vnm.train(
        start_date="2023-01-01",
        model_type=ModelType.CNN_LSTM,
        epoch_size=300,
        batch_size=32,
        time_step=4,
        ratio=0.8,
    )
    vnm.predict(predict_number=10)


if __name__ == "__main__":
    main()
