Train MAE Score: 0.2910974324 MAE
Test MAE Score: 0.7496318817 MAE
Train MAPE Score: 0.0054 MAPE
Test MAPE Score: 0.0142 MAPE
Train Score: 0.4331 RMSE
Test Score: 0.8812 RMSE
Train RMSPE: 0.0079 RMSPE
Test RMSPE: 0.0167 RMSPE
Time step: 4
Ratio: 0.8
Model type: ModelType.CNN_LSTM
Epoch size: 200
Batch size: 4
Date: from 2023-01-01 00:00:00 to 2023-07-05 09:31:04.661652
Model json: {"class_name": "Sequential", "config": {"name": "sequential_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 4, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_9_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_9", "trainable": true, "batch_input_shape": [null, 4, 2], "dtype": "float32", "filters": 32, "kernel_size": [3], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_31", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 300, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.10.0", "backend": "tensorflow"}