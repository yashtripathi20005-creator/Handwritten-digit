import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import MNISTNet

def load_model(model_path='mnist_model_best.pth'):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def preprocess_image(image_path, invert=False):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        invert: If True, invert colors (useful for white background with black digit)
    
    Returns:
        Tensor ready for model input
    """
    # Load image as grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Invert if needed (MNIST has black background with white digit)
    if invert:
        img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Normalize with MNIST statistics
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Convert to tensor
    img_tensor = transform(img_array).unsqueeze(0)
    
    return img_tensor, img_array

def predict_image(image_path, invert=False):
    """
    Predict the digit in a single image.
    
    Args:
        image_path: Path to the image file
        invert: If True, invert colors
    
    Returns:
        predicted_digit, confidence, probabilities
    """
    model, device = load_model()
    
    # Preprocess image
    img_tensor, img_array = preprocess_image(image_path, invert)
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    prediction, probabilities = model.predict(img_tensor)
    predicted_digit = prediction.item()
    confidence = probabilities[0][predicted_digit].item() * 100
    
    # Display image with prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Show original image
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title(f'Input Image')
    ax1.axis('off')
    
    # Show probabilities
    digits = range(10)
    probs = probabilities[0].cpu().numpy() * 100
    ax2.bar(digits, probs, color='skyblue')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability (%)')
    ax2.set_title(f'Prediction: {predicted_digit} (Confidence: {confidence:.1f}%)')
    ax2.set_xticks(digits)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.show()
    
    print(f'Predicted Digit: {predicted_digit}')
    print(f'Confidence: {confidence:.2f}%')
    print('\nProbabilities for each digit:')
    for i, prob in enumerate(probs):
        print(f'Digit {i}: {prob:.2f}%')
    
    return predicted_digit, confidence, probabilities

def predict_batch(image_paths, invert=False):
    """Predict multiple images."""
    model, device = load_model()
    
    fig, axes = plt.subplots(1, len(image_paths), figsize=(4*len(image_paths), 4))
    if len(image_paths) == 1:
        axes = [axes]
    
    for idx, (image_path, ax) in enumerate(zip(image_paths, axes)):
        img_tensor, img_array = preprocess_image(image_path, invert)
        img_tensor = img_tensor.to(device)
        
        prediction, probabilities = model.predict(img_tensor)
        predicted_digit = prediction.item()
        confidence = probabilities[0][predicted_digit].item() * 100
        
        ax.imshow(img_array, cmap='gray')
        ax.set_title(f'Pred: {predicted_digit}\nConf: {confidence:.1f}%')
        ax.axis('off')
        
        print(f'Image {idx+1}: Predicted {predicted_digit} (Confidence: {confidence:.2f}%)')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    print('=' * 50)
    print('MNIST Digit Recognition - Prediction Demo')
    print('=' * 50)
    print('\nTo use this script:')
    print('1. Save a digit image as "digit.png" in the current directory')
    print('2. Run: python predict.py')
    print('\nAlternatively, use in your own code:')
    print('from predict import predict_image')
    print('prediction, confidence, probs = predict_image("my_digit.png")')
    print('=' * 50)
    
    # Try to predict a sample from test data if no custom image provided
    try:
        from torchvision import datasets
        import random
        
        # Load a random test image
        test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        idx = random.randint(0, len(test_dataset)-1)
        image, label = test_dataset[idx]
        
        # Save temporarily
        from PIL import Image
        img = Image.fromarray(image.numpy(), mode='L')
        temp_path = 'temp_digit.png'
        img.save(temp_path)
        
        print(f'\nTesting with random MNIST digit (True label: {label})')
        predict_image(temp_path, invert=False)
        
        # Clean up
        import os
        os.remove(temp_path)
        
    except Exception as e:
        print(f'\nCould not load sample: {e}')
        print('Please provide your own image.')
