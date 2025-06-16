# Water_Quality_Prediction_RTSGAN-GNN-LSTM
Forecasting of water quality data (water temperature and dissolved oxygen) at three hydrological stations in Serbia. Additionally, RTSGAN is used to augment the dataset with synthetic data. It is then checked how and whether the additional data affect the prediction models (GNN and LSTM).

Firstly, missing data for DO at Senta and Zemun stations were populated using KNN. FInal datasets used for RTSGAN training are in the RTSGAN/data folder.

Secondly, to generate synthetic data, input training data for the RTSGAN needs to be prepared. To prepare data run prepare_data.ipynb script. After that run main_rtsgan.py. Both scripts are in the RTSGAN folder.
### CHECK ON PATHS IN THE FILES.

# Forecasting models - GNN and LSTM
Both models can be trained only with original data and additionally with synthtetic data. To train the models run the scripts in the GNN-LSTM folder. Again customize paths.
