import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # After 3 poolings: 28->14->7->3
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # Conv block 3: 7x7 -> 3x3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

    def predict(self, x):
        """Predict class probabilities for input images."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities
