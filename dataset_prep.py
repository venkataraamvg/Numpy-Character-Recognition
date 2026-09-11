import numpy as np
import csv

valid_classes = range(1, 36)
samples_per_class = 1000

# Tracking how many of each class we've collected
class_counts = {i: 0 for i in valid_classes}
x_data = []
y_data = []

print("Extracting images from CSV...")

#reading file line by line
with open('emnist-byclass-train.csv', 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        label = int(row[0])
        
        # If it's a class we need -> continue reading
        if label in class_counts and class_counts[label] < samples_per_class:
            y_data.append(label)
            
            # The remaining 784 columns are the pixel values (28x28 flattened)
            pixels = [int(p) for p in row[1:]]
            x_data.append(pixels)
            
            class_counts[label] += 1
            
        # stop reading after reaching the bthreshold
        if len(y_data) == len(valid_classes) * samples_per_class:
            print("Successfully collected 1000 samples for all 35 classes!")
            break

#convertion -> NumPy arrays
X = np.array(x_data, dtype=np.float32)
Y = np.array(y_data)

# normalization
X_normalized = X / 255.0

print(f"Data Shape: {X_normalized.shape}") 
print(f"Labels Shape: {Y.shape}") 

# Save to .npy (dont need to read csv again)
np.save('x_data.npy', X_normalized)
np.save('y_data.npy', Y)
print("Saved as x_data.npy and y_data.npy")