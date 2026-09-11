import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn

# 1 Helper function 
def get_char(index):
    #Conversion numbers -> char 
    if index < 9:
        return str(index + 1)
    else:
        return chr(index - 9 + ord('A'))

print("Loading test dataset and trained model...")
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
params = np.load('trained_params.npy', allow_pickle=True).item()

# 2. Run the test data through the trained network
A2_test, _ = nn.forward_propagation(X_test, params)

# Convert one-hot vectors -> standard label integers
y_pred = np.argmax(A2_test, axis=1)
y_true = np.argmax(Y_test, axis=1)

test_accuracy = np.mean(y_pred == y_true)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# 3.confution matrix generation
cm = np.zeros((35, 35), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix (35 Classes)')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("Saved 'confusion_matrix.png'")

# 4 analyze Incorrect Predictions
incorrect_indices = np.where(y_pred != y_true)[0]

# Select the first 5 incorrect predictions
selected_incorrect = incorrect_indices[:5]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, idx in enumerate(selected_incorrect):
    # Reshape the flattened 784 array back into a 28x28 image for viewing
    img = X_test[idx].reshape(28, 28)
    true_char = get_char(y_true[idx])
    pred_char = get_char(y_pred[idx])
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"True: {true_char}\nPred: {pred_char}")
    axes[i].axis('off')
#saving the incorrect prediction images
plt.tight_layout()
plt.savefig('incorrect_predictions.png')
print("Saved 'incorrect_predictions.png'")