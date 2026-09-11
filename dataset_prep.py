import numpy as np
import csv

# We need labels 1 through 35 (Digits 1-9 and Letters A-Z)
valid_classes = range(1, 36)
samples_per_class = 1000

# Track how many of each class we've collected
class_counts = {i: 0 for i in valid_classes}
x_data = []
y_data = []

print("Extracting images from CSV...")

# Read the file line by line to save memory
with open('emnist-byclass-train.csv', 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        label = int(row[0])
        
        # If it's a class we need, and we don't have 50 of them yet
        if label in class_counts and class_counts[label] < samples_per_class:
            y_data.append(label)
            
            # The remaining 784 columns are the pixel values (28x28 flattened)
            pixels = [int(p) for p in row[1:]]
            x_data.append(pixels)
            
            class_counts[label] += 1
            
        # Stop reading the 1.3GB file early once we have all 1750 samples
        if len(y_data) == len(valid_classes) * samples_per_class:
            print("Successfully collected 1000 samples for all 35 classes!")
            break

# Convert to NumPy arrays
X = np.array(x_data, dtype=np.float32)
Y = np.array(y_data)

# Normalize the pixel values to be between 0 and 1
X_normalized = X / 255.0

print(f"Data Shape: {X_normalized.shape}") 
print(f"Labels Shape: {Y.shape}") 

# Save them as lightweight NumPy files so you never have to read the CSV again
np.save('x_data.npy', X_normalized)
np.save('y_data.npy', Y)
print("Saved as x_data.npy and y_data.npy")