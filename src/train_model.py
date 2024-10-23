import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Define transformations for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pre-trained models
    ]),
    'validate': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set up directories
data_dir = 'C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')

# Create datasets
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'validate': datasets.ImageFolder(os.path.join(data_dir, 'validate'), data_transforms['validate'])
}

# Create data loaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'validate': DataLoader(image_datasets['validate'], batch_size=32, shuffle=False)
}

# Number of classes: garbage (1), clean (0)
class_names = image_datasets['train'].classes

# Use a pre-trained ResNet model and fine-tune it for this classification task
def get_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Modify the final fully connected layer to match our 2 classes
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# Train the model
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

if __name__ == "__main__":
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get the model and move it to the device
    model = get_model().to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), "C:/Users/Aleksandre Jelia/Desktop/CODE/garbage_detector/models/garbage_classifier.pth")
    print("Model training complete and saved!")
