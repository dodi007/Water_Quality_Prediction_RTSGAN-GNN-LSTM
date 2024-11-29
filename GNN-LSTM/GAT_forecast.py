import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
import pickle
import sys

sys.path.append(r'C:\Users\Ana\Desktop\GNNs\GAN-LSTM-GNN')

train_data = 'path_to_training_data'

df=pd.read_csv(train_data, parse_dates=['Date'])
dataset = df[['Protok_NS', 'Temperatura_NS','DO_NS','Protok_Z', 'Temperatura_Z','DO_Z','Protok_S', 'Temperatura_S','DO_S']]

test_data = 'path_to_test_data'

df=pd.read_csv(test_data, parse_dates=['Date'])
dataset_test = df[['Protok_NS', 'Temperatura_NS','DO_NS','Protok_Z', 'Temperatura_Z','DO_Z','Protok_S', 'Temperatura_S','DO_S']]

N=len(dataset)
N_test=len(dataset_test)

np.random.seed(0)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(dataset)
scaled_features1=scaled_features[:,6:9]
scaled_features2=scaled_features[:,0:3]
scaled_features3=scaled_features[:,3:6]

scaled_features_test=scaler.transform(dataset_test)
scaled_features1_test=scaled_features_test[:,6:9]
scaled_features2_test=scaled_features_test[:,0:3]
scaled_features3_test=scaled_features_test[:,3:6]


# Create the graph structure
adjacency_matrix = torch.tensor([[0, 0, 1],  # A -> C
                                 [0, 0, 1],  # B -> C
                                 [0, 0, 0]], dtype=torch.float32)

# Node features
node_features = torch.tensor([scaled_features1,
                              scaled_features2,
                              scaled_features3], dtype=torch.float32)

node_features_test = torch.tensor([scaled_features1_test,
                              scaled_features2_test,
                              scaled_features3_test], dtype=torch.float32)

edge_features = torch.tensor([[0.385,0.85],
                               [0.615,0.15]], dtype=torch.float32)

num_nodes=3
num_features=3
num_predictions=3
num_epochs=100
seq_length = 10  # Sequence length for each node's time series data
prediction_steps = 3  # Number of steps to predict ahead

train_size=N
# val_size= int(N_test * 0.4) 
test_size = N_test

#%%

data_list = []
data_list_batch=[]
batch_list=[]
batch_size=32
num=0
for i in range(train_size - seq_length-prediction_steps):
    edge_index = adjacency_matrix.nonzero().t()
    edge_attr = edge_features
    graph_data = Data(x=node_features[:, i:i + seq_length, :].reshape(num_nodes,num_features*seq_length),  
                      edge_index=edge_index, edge_attr=edge_attr, y=node_features[:, i+seq_length:i+ seq_length + prediction_steps,1:3])
    data_list.append(graph_data)
    data_list_batch.append(graph_data)
    num=num+1
    if num%batch_size==0:
        batch = Batch.from_data_list(data_list_batch)
        data_list_batch=[]
        batch_list.append(batch)
        
batch = Batch.from_data_list(data_list_batch)
batch_list.append(batch)

# set true if you want to use synthtetic sequences also
os.chdir(r'C:\Users\Ana\Desktop\GNNs\GAN\RTSGAN')
syn=False
if syn:
    for j in range(0,20):
        df_syn = pd.read_csv(f'best_syn_o2\sequence{j}.csv')
        dataset = df_syn[['Protok_NS', 'Temperatura_NS','DO_NS','Protok_Z', 'Temperatura_Z','DO_Z','Protok_S', 'Temperatura_S','DO_S']]
        N=len(dataset)
        scaled = scaler.transform(dataset)
        scaled_features1=scaled[:,6:9]
        scaled_features2=scaled[:,0:3]
        scaled_features3=scaled[:,3:6]
        node_features = torch.tensor([scaled_features1,
                                      scaled_features2,
                                      scaled_features3], dtype=torch.float32)
        data_list = []
        data_list_batch=[]
        for i in range(N - seq_length-prediction_steps):
            edge_index = adjacency_matrix.nonzero().t()
            edge_attr = edge_features
            graph_data = Data(x=node_features[:, i:i + seq_length, :].reshape(num_nodes,num_features*seq_length),  
                              edge_index=edge_index, edge_attr=edge_attr, y=node_features[:, i+seq_length:i+ seq_length + prediction_steps,1:3])
            data_list.append(graph_data)
            data_list_batch.append(graph_data)
            num=num+1
            if num%batch_size==0:
                batch = Batch.from_data_list(data_list_batch)
                data_list_batch=[]
                batch_list.append(batch)
        batch = Batch.from_data_list(data_list_batch)
        batch_list.append(batch)

#%%

# data_list = []
# for i in range(0,val_size-seq_length-prediction_steps):
#     edge_index = adjacency_matrix.nonzero().t()
#     edge_attr = edge_features
#     graph_data = Data(x=node_features_test[:, i:i + seq_length, :].reshape(num_nodes,num_features*seq_length),  
#                       edge_index=edge_index, edge_attr=edge_attr, y=node_features_test[:, i+seq_length:i+ seq_length + prediction_steps,1:3])
#     data_list.append(graph_data)
        
# batchval = Batch.from_data_list(data_list)

data_list = []
for i in range(0, N_test-seq_length-prediction_steps):
    edge_index = adjacency_matrix.nonzero().t()
    edge_attr = edge_features
    graph_data = Data(x=node_features_test[:, i:i + seq_length, :].reshape(num_nodes,num_features*seq_length),  
                      edge_index=edge_index, edge_attr=edge_attr, y=node_features_test[:, i+seq_length:i+ seq_length + prediction_steps,1:3])
    data_list.append(graph_data)
        
batchtest = Batch.from_data_list(data_list)


space = {
    'num_gat_layers': hp.choice('num_gat_layers', [2, 3]),
    'hidden_channels1': hp.choice('hidden_channels1', [32, 64, 128]),
    'hidden_channels2': hp.choice('hidden_channels2', [32, 64, 128]),
    'heads': hp.choice('heads', [2, 4, 8]),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'optimizer': hp.choice('optimizer', [
        {
            'type': 'adam',
            'lr': hp.loguniform('lr', -10, -1)
        },
        {
            'type': 'sgd',
            'lr': hp.loguniform('lr', -10, -1),
            'momentum': hp.uniform('momentum', 0, 1)
        }
    ])
}

#GAT Model
class GAT(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, params):
        self.params=params
        self.out_channels=out_channels
        super().__init__()
        self.conv1 = GATConv(in_channels=in_channels, out_channels=params['hidden_channels1'], heads=params['heads'], dropout=params['dropout'])
        if params['num_gat_layers']==2:
            self.conv2 = GATConv(params['hidden_channels1']*params['heads'], out_channels*2, heads=out_channels, dropout=params['dropout'],concat=False)
        if params['num_gat_layers']==3:
            self.conv2 = GATConv(params['hidden_channels1']*params['heads'], params['hidden_channels2'], heads=params['heads'], dropout=params['dropout'])
            self.conv3 = GATConv(params['hidden_channels2']*params['heads'], out_channels*2, heads=out_channels, dropout=params['dropout'],concat=False)


    def forward(self, x, edge_index, edge_weight):
        if self.params['num_gat_layers']==2:
            x = F.elu(self.conv1(x, edge_index, edge_attr=edge_weight))
            x = self.conv2(x, edge_index, edge_weight)
            x = x.view(x.shape[0],self.out_channels,2)
        if self.params['num_gat_layers']==3:
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            x = F.elu(self.conv2(x,edge_index, edge_weight))
            x = self.conv3(x, edge_index, edge_weight)
            x = x.view(x.shape[0],self.out_channels,2)
        return x
    
def objective(params):
    model = GAT(in_channels=num_features*seq_length, out_channels=prediction_steps, params=params)
    optimizer_params = params['optimizer']
    if optimizer_params['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['lr'])
    elif optimizer_params['type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_params['lr'], momentum=optimizer_params['momentum'])
    criterion = nn.MSELoss()
    num_epochs = 200
    best=1
    num=0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in batch_list:
            
            optimizer.zero_grad()
            output = model(data.x,data.edge_index,data.edge_attr)
            # output=output.reshape(batch_size*num_nodes, prediction_steps)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")
        # print(f"Epoch {epoch + 1}, Val_Loss: {val_loss:.6}")
    return {'loss': total_loss, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

best_params = space_eval(space, best)

print("Best hyperparameters: ", best_params)

with open('best_hyperparams_gnn.pkl', 'wb') as f:
    pickle.dump(best_params,f)

#%%

with open('best_hyperparams_gnn.pkl', 'rb') as f:
    loaded_best=pickle.load(f)
best_params=loaded_best
#%%

def best_gat(best_params):
    train_loss=[]
    model = GAT(in_channels=num_features*seq_length, out_channels=prediction_steps, params=best_params)
    optimizer = best_params['optimizer']
    
    if optimizer['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer['lr'])
    elif optimizer['type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer['lr'], momentum=optimizer['momentum'])
    criterion = nn.MSELoss()
    num_epochs = 200
    best=1
    num=0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in batch_list:
            
            optimizer.zero_grad()
            output = model(data.x,data.edge_index,data.edge_attr)
            # output=output.reshape(batch_size*num_nodes, prediction_steps)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # losses.append(total_loss)
        train_loss.append(total_loss)


        print(f"Epoch {epoch + 1}, Loss: {total_loss:.6f}")
       
    return model,train_loss

model,train_loss = best_gat(best_params)

torch.save(model, 'gnnmodel_with_o2_syn.pth')


#model=torch.load('gnnmodel_with_o2_best.pth')


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2)


model.eval()  # Set the model to evaluation mode
y_pred1=[]
y_pred2=[]
y_pred3=[]
y_true1=[]
y_true2=[]
y_true3=[]

with torch.no_grad():
    output = model(batchtest.x, batchtest.edge_index, batchtest.edge_attr)
    output = output.view(-1, 3, 2)  # Assuming each node outputs two values
    batchtest.y = batchtest.y.view(-1,3,2)
    for i in range(0,len(output),3):
        
        y_pred1.append(output[i,:,:].detach().numpy())
        y_pred2.append(output[i+1,:,:].detach().numpy())
        y_pred3.append(output[i+2,:,:].detach().numpy())
        y_true1.append(batchtest.y[i,:,:].detach().numpy())
        y_true2.append(batchtest.y[i+1,:,:].detach().numpy())
        y_true3.append(batchtest.y[i+2,:,:].detach().numpy())

y_pred1=np.array(y_pred1)
y_pred2=np.array(y_pred2)
y_pred3=np.array(y_pred3)
y_true1=np.array(y_true1)
y_true2=np.array(y_true2)
y_true3=np.array(y_true3)

n_parameters=2
# Calculate metrics for each output
stations=['Senta', 'Novi Sad', 'Zemun']
parameters=['Temperature','DO']
days=['1st','2nd','3rd']

for i in range(n_parameters):
    for j in range(3):
        rmse_score1 = rmse(y_true1[:, j, i], y_pred1[:, j, i])
        r2_score_value1 = r2_score(y_true1[:, j, i], y_pred1[:, j, i])
        nse_score1 = nse(y_true1[:, j, i].ravel(), y_pred1[:, j, i].ravel())

        print(f"Scores for {parameters[i]} for {days[j]} day at Senta- RMSE: {rmse_score1}, R2 Score: {r2_score_value1}, NSE: {nse_score1}")
for i in range(n_parameters):
    for j in range(3):
        rmse_score1 = rmse(y_true2[:, j, i], y_pred2[:, j, i])
        r2_score_value1 = r2_score(y_true2[:, j, i], y_pred2[:, j, i])
        nse_score1 = nse(y_true2[:, j, i].ravel(), y_pred2[:, j, i].ravel())

        print(f"Scores for {parameters[i]} for {days[j]} day at Novi Sad- RMSE: {rmse_score1}, R2 Score: {r2_score_value1}, NSE: {nse_score1}")

for i in range(n_parameters): 
    for j in range(3):
        rmse_score1 = rmse(y_true3[:, j, i], y_pred3[:, j, i])
        r2_score_value1 = r2_score(y_true3[:, j, i], y_pred3[:, j, i])
        nse_score1 = nse(y_true3[:, j, i].ravel(), y_pred3[:, j, i].ravel())

        print(f"Scores for {parameters[i]} for {days[j]} day at Zemun- RMSE: {rmse_score1}, R2 Score: {r2_score_value1}, NSE: {nse_score1}")


results_df = pd.DataFrame(columns=['Station', 'Parameter', 'Day', 'RMSE', 'R2 Score', 'NSE'])

stations=['Senta', 'Novi Sad', 'Zemun']
parameters = ['Temperature','DO']
days = ['1st', '2nd', '3rd']

all_y_trues = [y_true1, y_true2, y_true3]
all_y_preds = [y_pred1, y_pred2, y_pred3]

for station_idx, station in enumerate(stations):
    y_true = all_y_trues[station_idx]
    y_pred = all_y_preds[station_idx]

    for param_idx, parameter in enumerate(parameters):
        for day_idx, day in enumerate(days):
            rmse_score = rmse(y_true[:, day_idx, param_idx], y_pred[:, day_idx, param_idx])
            r2_score_value = r2_score(y_true[:, day_idx, param_idx], y_pred[:, day_idx, param_idx])
            nse_score = nse(y_true[:, day_idx, param_idx].ravel(), y_pred[:, day_idx, param_idx].ravel())

            results_df = results_df.append({
                'Station': station,
                'Parameter': parameter,
                'Day': day,
                'RMSE': rmse_score,
                'R2 Score': r2_score_value,
                'NSE': nse_score
            }, ignore_index=True)