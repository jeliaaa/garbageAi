import torch
from torchvision import models, transforms
from PIL import Image
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from playsound import playsound

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
def load_model(path):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

# Detect if the image contains garbage or not
def detect_garbage(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension

    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    
    return preds.item()  # Return the prediction index

if __name__ == "__main__":
    # Use Tkinter to open a file dialog
    Tk().withdraw()  # Hide the root window
    image_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if image_path:  # Proceed if an image was selected
        # Load the trained model
        model = load_model("C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/models/garbage_classifier.pth")

        # Detect garbage
        prediction = detect_garbage(model, image_path)
        
        if prediction == 1:  # If garbage is detected
            print("Garbage detected! Playing alarm...")
            playsound("C:/Users/Aleksandre Jelia/Desktop/CODE/netDetection/sounds/alarm.mp3")  # Update with your MP3 file path
        else:
            print("The lake is clean.")
    else:
        print("No image selected.")
