import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import train_test_split
import os
import sys
from tensorflow.keras.callbacks import EarlyStopping

#here you need to chose for which station you want to train LSTM
station_ns = ['Protok_NS', 'Temperatura_NS','DO_NS']
station_s = ['Protok_S', 'Temperatura_S','DO_S']
station_z = ['Protok_Z', 'Temperatura_Z','DO_Z']

train_data = 'path_to_training_data'

df=pd.read_csv(train_data, parse_dates=['Date'])
   
dataset = df[station_ns]

test_data = 'path_to_test_data'

df=pd.read_csv(test_data, parse_dates=['Date'])

dataset_test = df[station_ns]


np.random.seed(0)

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(dataset)
scaled_features_test=scaler.transform(dataset_test)
# Prepare the sequences
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in), :])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out), 1:3])
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 10, 3

X_train, y_train = create_sequences(scaled_features, n_steps_in, n_steps_out)
X_test, y_test = create_sequences(scaled_features_test, n_steps_in, n_steps_out)

# set true if you want to use synthtetic sequences also

os.chdir(r'C:\Users\Ana\Desktop\GNNs\GAN\RTSGAN')
syn=False
if syn:
    for i in range(0,8):
        df_syn = pd.read_csv(f'best_syn_o2\sequence{i}.csv')
        dataset = df_syn[['Protok_NS', 'Temperatura_NS','DO_NS']]
        scaled = scaler.transform(dataset)
        X_s,y_s=create_sequences(scaled,n_steps_in,n_steps_out)
        X_train = np.concatenate((X_train, X_s), axis=0) 
        y_train = np.concatenate((y_train, y_s), axis=0) 
    
N_test=len(scaled_features_test)

test_size = N_test

space = {
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'units_layer_1': hp.choice('units_layer_1', [32, 64, 128]),
    'units_layer_2': hp.choice('units_layer_2', [32, 64, 128]),
    'units_layer_3': hp.choice('units_layer_3', [32, 64, 128]),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'activation_layer_1': hp.choice('activation_layer_1', ['relu', 'sigmoid', 'tanh']),
    'activation_layer_2': hp.choice('activation_layer_2', ['relu', 'sigmoid', 'tanh']),
    'activation_layer_3': hp.choice('activation_layer_3', ['relu', 'sigmoid', 'tanh']),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
    'optimizer': hp.choice('optimizer', [
        {
            'type': 'adam',
            'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01))
        },
        {
            'type': 'sgd',
            'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
            'momentum': hp.uniform('momentum', 0, 1)
        }]),
    'batch_size': hp.choice('batch_size',[16,32,64])
    
}

#LSTM model
def lstm_model(params):
    model = Sequential()
    
    # Add multiple LSTM layers
    for i in range(params['num_layers']):
        # Choose the number of units based on the layer
        units = params[f'units_layer_{i+1}']
        activation = params[f'activation_layer_{i+1}']
        # Return sequences for all layers but the last one
        return_sequences = i < params['num_layers'] - 1
        model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=(X_train.shape[1], X_train.shape[2]),activation=activation))
        model.add(Dropout(params['dropout']))

    model.add(Dense(n_steps_out*2, activation=params['activation']))  # 2 outputs for each day ahead
    model.add(Reshape((n_steps_out, 2)))

    optimizer_params = params['optimizer']
    if optimizer_params['type'] == 'adam':
        optimizer = Adam(learning_rate=optimizer_params['lr'])
    else:  # SGD
        optimizer = SGD(learning_rate=optimizer_params['lr'], momentum=optimizer_params['momentum'])

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.001, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=200, batch_size=params['batch_size'], verbose=2)
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=params['batch_size'], callbacks=[early_stopping], verbose=2)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=2)
    return {'loss': loss, 'status': STATUS_OK}

# Run the optimization
trials = Trials()
best = fmin(fn=lstm_model, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

best_params = space_eval(space, best)

print("Best hyperparameters: ", best_params)

with open('best_hyperparams_lstm_ns.pkl', 'wb') as f:
    pickle.dump(best_params,f)


with open('best_hyperparams_lstm_ns.pkl', 'rb') as f:
    loaded_best=pickle.load(f)

best_params=loaded_best



def best_lstm(params):
    
    model = Sequential()
    
    # Add multiple LSTM layers
    for i in range(params['num_layers']):
        # Choose the number of units based on the layer
        unitss = params[f'units_layer_{i+1}']
        activationn = params[f'activation_layer_{i+1}']
        # Return sequences for all layers but the last one
        return_sequences = i < params['num_layers'] - 1
        model.add(LSTM(units=unitss, return_sequences=return_sequences, input_shape=(X_train.shape[1], X_train.shape[2]),activation=activationn))
        model.add(Dropout(params['dropout']))

    model.add(Dense(n_steps_out*2, activation=params['activation']))  # 2 outputs for each day ahead
    model.add(Reshape((n_steps_out,2)))

    optimizer = params['optimizer']
    if optimizer['type'] == 'adam':
        optimizer = Adam(learning_rate=optimizer['lr'])
    else:  
        optimizer = SGD(learning_rate=optimizer['lr'], momentum=optimizer['momentum'])

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001, restore_best_weights=True)
    
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=batch_size[params['batch_size']], callbacks=[early_stopping], verbose=2)
    history=model.fit(X_train, y_train, epochs=200, batch_size=params['batch_size'], verbose=2)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=2)
    return model,history.history['loss']


model,loss = best_lstm(best)
model.save('model_lstm_ns')

loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

#model=tf.keras.models.load_model('model_lstm_ns')

y_pred = model.predict(X_test)
y_true=y_test

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2)
#%%
# Calculate metrics for each output
parameter=['Temperature','DO']
day=['1st','2nd','3rd']
for i in range(2):  # Assuming you have 2 outputs
    for j in range(3):
        rmse_score = rmse(y_true[:, j, i], y_pred[:, j, i])
        r2_score_value = r2_score(y_true[:, j, i], y_pred[:, j, i])
        nse_score = nse(y_true[:, j, i].ravel(), y_pred[:, j, i].ravel())

        print(f"Scores for {parameter[i]} for {day[j]} day - RMSE: {rmse_score}, R2 Score: {r2_score_value}, NSE: {nse_score}")


results_df = pd.DataFrame(columns=['Station', 'Parameter', 'Day', 'RMSE', 'R2 Score', 'NSE'])

stations = ['Zemun']
parameters = ['Temperature','DO']
days = ['1st', '2nd', '3rd']

# Assuming y_pred1, y_pred2, y_pred3, y_true1, y_true2, y_true3 are defined as per your previous code

all_y_trues = [y_true]
all_y_preds = [y_pred]

for station_idx, station in enumerate(stations):
    y_true = all_y_trues[station_idx]
    y_pred = all_y_preds[station_idx]

    for param_idx, parameter in enumerate(parameters):
        for day_idx, day in enumerate(days):
            rmse_score = rmse(y_true[:, day_idx, param_idx], y_pred[:, day_idx, param_idx])
            r2_score_value = r2_score(y_true[:, day_idx, param_idx], y_pred[:, day_idx, param_idx])
            nse_score = nse(y_true[:, day_idx, param_idx].ravel(), y_pred[:, day_idx, param_idx].ravel())

            # Append row to DataFrame
            results_df = results_df.append({
                'Station': station,
                'Parameter': parameter,
                'Day': day,
                'RMSE': rmse_score,
                'R2 Score': r2_score_value,
                'NSE': nse_score
            }, ignore_index=True)