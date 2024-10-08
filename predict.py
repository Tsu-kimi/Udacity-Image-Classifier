import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models

def get_input_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Image prediction script")
    parser.add_argument('--image', type=str, required=True, help='Path to the image for prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top K classes to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File with category names')
    parser.add_argument('--gpu', type=str, default='gpu', help='Use GPU for inference if available')

    return parser.parse_args()

def load_saved_model(checkpoint_path):
    """Load the saved model checkpoint and rebuild the model."""
    checkpoint = torch.load(checkpoint_path)

    # Load a pretrained model and modify it to match the saved checkpoint
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    # Ensure the parameters are not updated during training
    for param in model.parameters():
        param.requires_grad = False

    return model

def process_image(image_path):
    """Preprocess an image for the model to use for inference."""
    img = PIL.Image.open(image_path)
    
    # Resize the image, maintaining aspect ratio
    original_width, original_height = img.size
    if original_width < original_height:
        img.thumbnail((256, int(256 * original_height / original_width)))
    else:
        img.thumbnail((int(256 * original_width / original_height), 256))

    # Crop the center 224x224 of the image
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Normalize and transpose image to match model expectations
    np_img = np.array(img) / 255.0
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - means) / stds
    np_img = np_img.transpose((2, 0, 1))

    return np_img

def predict(image_path, model, device, cat_to_name, top_k=5):
    """Make predictions on the image and return the top K probabilities and corresponding classes."""
    model.to(device)
    model.eval()
    
    # Preprocess image and convert to tensor
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Perform inference
        output = model.forward(image_tensor)
        probabilities = torch.exp(output)
    
    # Get the top K probabilities and indices
    top_probs, top_indices = probabilities.topk(top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    top_flowers = [cat_to_name[str(cls)] for cls in top_classes]
    
    return top_probs, top_flowers

def display_predictions(probs, flowers):
    """Print the top predicted flowers and their probabilities."""
    for i, (flower, prob) in enumerate(zip(flowers, probs)):
        print(f"Rank {i+1}: Flower = {flower}, Likelihood = {ceil(prob * 100)}%")

def main():
    """Main function to run the script."""
    # Get input arguments
    args = get_input_args()

    # Load category to name mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the model from checkpoint
    model = load_saved_model(args.checkpoint)
    
    # Determine if GPU should be used
    device = check_gpu(args.gpu)
    
    # Perform prediction
    top_probs, top_flowers = predict(args.image, model, device, cat_to_name, args.top_k)
    
    # Display the predictions
    display_predictions(top_probs, top_flowers)

if __name__ == "__main__":
    main()
