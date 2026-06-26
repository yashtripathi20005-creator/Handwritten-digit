import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model import MNISTNet

def load_model(model_path='mnist_model_best.pth'):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def evaluate_model():
    """Evaluate the trained model on test data."""
    
    # Load model
    model, device = load_model('mnist_model_best.pth')
    print(f'Model loaded successfully!')
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print('Confusion matrix saved as confusion_matrix.png')
    
    # Display some predictions
    display_predictions(model, device)

def display_predictions(model, device, num_samples=10):
    """Display random test samples with predictions."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Random sample
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            image, label = test_dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            prediction, probabilities = model.predict(image_tensor)
            
            ax.imshow(image.squeeze(), cmap='gray')
            predicted_digit = prediction.item()
            confidence = probabilities[0][predicted_digit].item() * 100
            ax.set_title(f'True: {label}\nPred: {predicted_digit}\nConf: {confidence:.1f}%')
            ax.axis('off')
            
            # Color border based on correctness
            if predicted_digit == label:
                ax.spines['bottom'].set_color('green')
                ax.spines['top'].set_color('green')
                ax.spines['left'].set_color('green')
                ax.spines['right'].set_color('green')
            else:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['top'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()
    print('Sample predictions saved as sample_predictions.png')

if __name__ == '__main__':
    evaluate_model()
