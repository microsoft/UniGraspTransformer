import numpy as np

scaler_path =f'./utils/autoencoding_ours/scaler.npy'
scaler = np.load(scaler_path, allow_pickle=True).item()
BS = 100 
X = np.random.randn(BS,64)  # [BS, 64]

# scaler.fit_transform() : learn mean & var on and transform input data
# scaler.transform() : transform input data using learned mean & var
X_normalized = scaler.transform(X) # [BS, 64]
pass