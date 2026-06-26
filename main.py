"""
MNIST Handwritten Digit Recognizer
==================================
A complete pipeline for training, evaluating, and predicting
handwritten digits using a Convolutional Neural Network.

Usage:
    python main.py train    - Train the model
    python main.py evaluate - Evaluate the trained model
    python main.py predict  - Make predictions on custom images
    python main.py all      - Run everything (train, evaluate, show predictions)
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='MNIST Handwritten Digit Recognizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train       # Train the model
  python main.py evaluate    # Evaluate the model
  python main.py predict     # Make predictions
  python main.py all         # Run full pipeline
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='all',
        choices=['train', 'evaluate', 'predict', 'all'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('MNIST HANDWRITTEN DIGIT RECOGNIZER')
    print('=' * 60)
    
    if args.command == 'train' or args.command == 'all':
        print('\n[1] Training the model...')
        print('-' * 60)
        from train import train_model
        train_model()
    
    if args.command == 'evaluate' or args.command == 'all':
        print('\n[2] Evaluating the model...')
        print('-' * 60)
        from evaluate import evaluate_model
        evaluate_model()
    
    if args.command == 'predict' or args.command == 'all':
        print('\n[3] Prediction demo...')
        print('-' * 60)
        from predict import predict_image
        print('To predict your own images:')
        print('  from predict import predict_image')
        print('  predict_image("path/to/your/image.png", invert=True)')
        print('\nRunning sample prediction...')
        
        # Load a random sample
        try:
            from torchvision import datasets
            from PIL import Image
            import random
            
            test_dataset = datasets.MNIST(root='./data', train=False, download=True)
            idx = random.randint(0, len(test_dataset)-1)
            image, label = test_dataset[idx]
            img = Image.fromarray(image.numpy(), mode='L')
            temp_path = 'sample_digit.png'
            img.save(temp_path)
            
            print(f'\nSample digit (True label: {label})')
            predict_image(temp_path, invert=False)
            
            os.remove(temp_path)
        except Exception as e:
            print(f'Error in prediction demo: {e}')
    
    print('\n' + '=' * 60)
    print('Done! Check the generated files:')
    print('  - mnist_model_best.pth (best model)')
    print('  - mnist_model_final.pth (final model)')
    print('  - confusion_matrix.png')
    print('  - sample_predictions.png')
    print('=' * 60)

if __name__ == '__main__':
    main()
