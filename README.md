# Water_Quality_Prediction_RTSGAN-GNN-LSTM
Forecasting of water quality data (water temperature and dissolved oxygen) at three hydrological stations in Serbia. Additionally, RTSGAN is used to augment the dataset with synthetic data. It is then checked how and whether the additional data affect the prediction models (GNN and LSTM).
## 🚀 Key Features
- **RTSGAN**: Synthetic data generation to augment limited hydrological datasets
- **GNN**: Spatial-temporal modeling using river network dependencies
- **LSTM**: Temporal dynamics modeling with consistent downstream performance
- Performance evaluated using RMSE and R² across multiple prediction days

## 📦 Setup Instructions

To run the project, we recommend using a Python virtual environment with Python 3.8 or newer installed.

```bash
git clone https://github.com/dodi007/Water_Quality_Prediction_RTSGAN-GNN-LSTM.git
cd Water_Quality_Prediction_RTSGAN-GNN-LSTM
```
## Install Required Packages
```bash
pip install -r requirements.txt
```
Firstly, missing data for DO at Senta and Zemun stations were populated using KNN. Run the script from KNN_for_missing_values_imputation folder. FInal, populated datasets used for RTSGAN training are in the RTSGAN/data folder.
The original/raw data can be found in the raw_data folder.

Secondly, to generate synthetic data, input training data for the RTSGAN needs to be prepared. To prepare data run prepare_data.ipynb script from the RSTGAN folder. After that run main_rtsgan.py.
### CHECK ON PATHS IN THE FILES.

## Forecasting models - GNN and LSTM
Both DL models can be trained only with original data and additionally with synthtetic data. To train the models run the scripts in the GNN-LSTM folder. Again customize paths and portion of the synthetic data.
