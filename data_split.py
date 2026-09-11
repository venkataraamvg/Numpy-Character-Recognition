import numpy as np

# 1. Load the generated dataset
X = np.load('x_data.npy')
Y = np.load('y_data.npy')
num_samples = X.shape[0]

print(f"Loaded {num_samples} total images.")

# 2. Shuffle the dataset randomly
# We generate a list of random indices and shuffle both X and Y in the exact same order
np.random.seed(42) # Set seed for reproducibility
shuffled_indices = np.random.permutation(num_samples)
X_shuffled = X[shuffled_indices]
Y_shuffled = Y[shuffled_indices]

# 3. Convert labels from (1-35) to (0-34) for zero-indexed arrays
Y_adjusted = Y_shuffled - 1

# 4. Apply One-Hot Encoding for 35 classes
# This turns a label like '2' into [0, 0, 1, 0, ..., 0]
Y_one_hot = np.eye(35)[Y_adjusted]

# 5. Define split ratios (70% Train, 15% Validation, 15% Test)
train_size = int(num_samples * 0.70)
val_size = int(num_samples * 0.15)

# 6. Slice the arrays into their final sets
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