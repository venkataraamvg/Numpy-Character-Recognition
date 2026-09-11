import numpy as np

# 1. Loading dataset
X = np.load('x_data.npy')
Y = np.load('y_data.npy')
num_samples = X.shape[0]

print(f"Loaded {num_samples} total images.")

# 2. Shuffle the dataset randomly
np.random.seed(42) 
shuffled_indices = np.random.permutation(num_samples)
X_shuffled = X[shuffled_indices]
Y_shuffled = Y[shuffled_indices]

# 3. Coversion (1-35) -> (0-34) 
Y_adjusted = Y_shuffled - 1

# 4. Appling One-Hot Encoding
Y_one_hot = np.eye(35)[Y_adjusted]

# 5. Splitting
train_size = int(num_samples * 0.70)
val_size = int(num_samples * 0.15)

# 6. Slicing
X_train = X_shuffled[:train_size]
Y_train = Y_one_hot[:train_size]

X_val = X_shuffled[train_size:train_size + val_size]
Y_val = Y_one_hot[train_size:train_size + val_size]

X_test = X_shuffled[train_size + val_size:]
Y_test = Y_one_hot[train_size + val_size:]

print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Validation data shape: X={X_val.shape}, Y={Y_val.shape}")
print(f"Testing data shape: X={X_test.shape}, Y={Y_test.shape}")

# 7. Save the final sets
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('X_val.npy', X_val)
np.save('Y_val.npy', Y_val)
np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)

print("Data successfully split and saved. Data Preparation Phase Complete.")