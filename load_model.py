"""
Example script for loading and using the trained ResNet18 jamming detector model.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


def load_model(model_path='resnet18_jamming_detector.pth', device=None):
    """
    Load the trained model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on (default: auto-detect)
    
    Returns:
        model: Loaded PyTorch model
        checkpoint: Dictionary containing model metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Method 1: Load the full model (easiest)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    
    # Method 2: If you need to recreate the architecture manually:
    # resnet = models.resnet18(pretrained=True)
    # for param in resnet.parameters():
    #     param.requires_grad = False
    # resnet.fc = nn.Sequential(
    #     nn.Linear(resnet.fc.in_features, 128),
    #     nn.ReLU(),
    #     nn.Dropout(0.65),
    #     nn.Linear(128, 1),
    #     nn.Sigmoid()
    # )
    # resnet.load_state_dict(checkpoint['model_state_dict'])
    # resnet = resnet.to(device)
    # resnet.eval()
    # model = resnet
    
    print(f"Model loaded successfully!")
    print(f"Final Validation Accuracy: {checkpoint['final_val_accuracy']:.2f}%")
    print(f"Best Validation Accuracy: {checkpoint['best_val_accuracy']:.2f}% (Epoch {checkpoint['best_epoch']})")
    
    return model, checkpoint


def preprocess_image(image_path):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3-channel input
        transforms.Resize((224, 224)),  # ResNet18 default
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                           [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict_image(model, image_path, device=None, threshold=0.5):
    """
    Predict whether an image contains jamming or not.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the image file
        device: Device to run inference on
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        prediction: 'jamming' or 'no_jamming'
        confidence: Confidence score (0-1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        confidence = output.item()
        prediction = 'jamming' if confidence > threshold else 'no_jamming'
    
    return prediction, confidence


def predict_batch(model, image_paths, device=None, threshold=0.5):
    """
    Predict on multiple images at once.
    
    Args:
        model: Trained PyTorch model
        image_paths: List of image file paths
        device: Device to run inference on
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        predictions: List of ('jamming' or 'no_jamming', confidence) tuples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)
    
    batch_tensor = torch.stack(images).to(device)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        predictions = []
        for output in outputs:
            confidence = output.item()
            prediction = 'jamming' if confidence > threshold else 'no_jamming'
            predictions.append((prediction, confidence))
    
    return predictions


# Example usage
if __name__ == "__main__":
    # Load the model
    model_path = 'resnet18_jamming_detector.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train_model_resnet18.py")
    else:
        model, checkpoint = load_model(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Example 1: Predict on a single image
        print("\n" + "="*60)
        print("Example: Single Image Prediction")
        print("="*60)
        
        # Try to find a test image
        test_image = None
        if os.path.exists("data/val/jamming"):
            test_files = os.listdir("data/val/jamming")
            if test_files:
                test_image = os.path.join("data/val/jamming", test_files[0])
        elif os.path.exists("data/val/no_jamming"):
            test_files = os.listdir("data/val/no_jamming")
            if test_files:
                test_image = os.path.join("data/val/no_jamming", test_files[0])
        
        if test_image:
            prediction, confidence = predict_image(model, test_image, device)
            print(f"Image: {test_image}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f} ({'jamming' if confidence > 0.5 else 'no_jamming'})")
        else:
            print("No test images found. To test, provide an image path:")
            print("  prediction, confidence = predict_image(model, 'path/to/image.png', device)")
        
        # Example 2: Batch prediction
        print("\n" + "="*60)
        print("Example: Batch Prediction")
        print("="*60)
        print("To predict on multiple images:")
        print("  image_paths = ['image1.png', 'image2.png', ...]")
        print("  predictions = predict_batch(model, image_paths, device)")
        print("  for (pred, conf) in predictions:")
        print("      print(f'{pred}: {conf:.4f}')")

