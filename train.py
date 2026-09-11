import numpy as np
import neural_network as nn 

def calculate_accuracy(predictions, labels):
    
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    return np.mean(pred_classes == true_classes)

def train_model(X_train, Y_train, X_val, Y_val, epochs=500, learning_rate=0.5):

    #Executes the training and validation loops over a set number of epochs.
    
    
    # 1. Initialize weights and biases
    params = nn.init_params(input_size=784, hidden_size=128, output_size=35)
    
    # 2. Setup dictionary to document metrics properly
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        # Drop the learning rate by 10% every 500 epochs
        if epoch > 0 and epoch % 500 == 0:
            learning_rate = learning_rate * 0.9 
            print(f"Learning rate decayed to: {learning_rate:.5f}")
        
        # Forward propagation
        A2_train, cache = nn.forward_propagation(X_train, params)
        
        # Cross-entropy loss and accuracy
        train_loss = nn.compute_loss(Y_train, A2_train)
        train_acc = calculate_accuracy(A2_train, Y_train)
        
        # backpropagation
        grads = nn.back_propagation(X_train, Y_train, params, cache)
        
        # Gradient descent (Update parameters)
        params = nn.update_parameters(params, grads, learning_rate)
        
        # Forward propagation ONLY (No backpropagation or weight updates)
        A2_val, _ = nn.forward_propagation(X_val, params)
        val_loss = nn.compute_loss(Y_val, A2_val)
        val_acc = calculate_accuracy(A2_val, Y_val)
        
        # documentation
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Print the progress every 50 eppoch
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
    print("Training Complete!")
    return params, history

if __name__ == "__main__":
    print("Loading datasets...")
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    X_val = np.load('X_val.npy')
    Y_val = np.load('Y_val.npy')
    
    # Execute the model
    trained_params, metrics_history = train_model(X_train, Y_train, X_val, Y_val, epochs=500, learning_rate=0.5)
    
    # Save the trained weights to disk 
    print("Final model parameters successfully saved to 'trained_params.npy'.")