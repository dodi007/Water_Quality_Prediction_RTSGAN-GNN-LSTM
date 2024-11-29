import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read the CSV files
df1 = pd.read_csv('NSKNN.csv', parse_dates=['Date'])
df2 = pd.read_csv('ZEMUNpresekdatumaDO.csv', parse_dates=['Date'])
df3 = pd.read_csv('SENTApresekdatumaDO.csv', parse_dates=['Date'])

# Assuming station1 has the most complete data
# Align station2 and station3 data to station1 using Date
df2 = df1[['Date']].merge(df2, on='Date', how='left')
df3 = df1[['Date']].merge(df3, on='Date', how='left')

# Combine data from all stations
combined_df = df1.merge(df2, on='Date', how='left', suffixes=('', '_2'))
combined_df = combined_df.merge(df3, on='Date', how='left', suffixes=('', '_3'))

# Fill missing DO values with a placeholder
combined_df.fillna({'DO_2': -999, 'DO_3': -999}, inplace=True)

# Feature Engineering: Adding day of year and month as features
combined_df['DayOfYear'] = combined_df['Date'].dt.dayofyear
combined_df['Month'] = combined_df['Date'].dt.month


nan_values = combined_df.isna()

    # Iterate over the DataFrame to find indices of NaN values
nan_locations = []
for column in nan_values.columns:
    rows_with_nan = nan_values.index[nan_values[column]].tolist()
    if rows_with_nan:
        nan_locations.extend([(column, row) for row in rows_with_nan])


# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_df[['Flow', 'Temperature','DO','Flow_2', 'Temperature_2','Flow_3', 'Temperature_3', 'DayOfYear', 'Month']])
combined_df[['Flow_scaled', 'Temperature_scaled','DO_scaled','Flow_2_scaled','Temperature_2_scaled','Flow_3_scaled','Temperature_3_scaled', 'DayOfYear_scaled', 'Month_scaled']] = scaled_features

def knn_impute_with_resampling(df, target_col, k, n_iterations=100):
    missing_indices = df[df[target_col] == -999].index
    imputed_values = np.zeros((len(missing_indices), 100))  # 100 for the number of bootstrap samples

    for i in range(100):
        # Create bootstrap sample
        sample = resample(df)
        known = sample[sample[target_col] != -999]
        unknown = sample[sample[target_col] == -999]
        
        # Train kNN regressor
        features = ['Flow_scaled', 'Temperature_scaled','DO_scaled','Flow_2_scaled','Temperature_2_scaled','Flow_3_scaled','Temperature_3_scaled', 'DayOfYear_scaled', 'Month_scaled']
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(known[features], known[target_col])
        
        # Predict missing values
        imputed = knn.predict(df.loc[missing_indices][features])
        imputed_values[:, i] = imputed
    
    # Aggregate imputed values
    final_imputed = np.mean(imputed_values, axis=1)
    df.loc[missing_indices, target_col] = final_imputed


# Apply the function for DO columns of station2 and station3
knn_impute_with_resampling(combined_df,'DO_2',k=7)
knn_impute_with_resampling(combined_df,'DO_3',k=7)

# Extract the imputed DO values for station2 and station3
station2_daily_do = combined_df[['Date', 'DO_2']]
station3_daily_do = combined_df[['Date', 'DO_3']]

# Save or process the imputed data as needed
station2_daily_do.to_csv('zemun_daily_do_new.csv', index=False)
station3_daily_do.to_csv('senta_daily_do_new.csv', index=False)
