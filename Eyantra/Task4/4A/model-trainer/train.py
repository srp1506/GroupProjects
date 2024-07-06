'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			GG_3895
# Author List:		Ashwin Agrawal, Siddhant Godbole, Soham Pawar, Aditya Waghmare
# Filename:			task_2b_model_training.py
###################################################################################################


import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# torch.cuda.set_per_process_memory_fraction(0.5)
# Define a preprocessing function for training images
image_size = 400
num_epochs = 20
b_size = 8
scale_ = 0.8
model_name = 'efficientnet-b4' #b0 to b7

def preprocess_train_image(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(scale_, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47024060015494107, 0.43551567841187444, 0.409760019835011], std=[0.24909021912996998, 0.2409229164701201, 0.24159483373089383]),
    ])
    
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)

# Define a preprocessing function for testing images
def preprocess_test_image(image):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize both width and height to 256
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)


# Path to the root folder containing class folders for training dataset
train_data_dir = 'dataset'
# Path to the root folder containing class folders for test dataset
test_data_dir = 'test_dataset'

# Map the class labels to their corresponding folder names
class_to_label = {
    "combat": 0,
    "humanitarianaid": 1,
    "militaryvehicles": 2,
    "fire": 3,
    "destroyedbuilding": 4
}

# Create a custom dataset using ImageFolder and apply the class_to_label mapping for training dataset
custom_dataset = ImageFolder(root=train_data_dir, transform=preprocess_train_image)

# Replace the class labels in the dataset with numerical labels using the mapping
modified_samples = []
for item in custom_dataset.samples:
    path, label = item
    folder_name = os.path.basename(os.path.dirname(path))
    numerical_label = class_to_label.get(folder_name, -1)
    if numerical_label != -1:
        modified_samples.append((path, numerical_label))

# Update the dataset samples with the modified list for training dataset
custom_dataset.samples = modified_samples

# Create DataLoader for training dataset
train_loader = DataLoader(custom_dataset, batch_size=b_size, shuffle=True)

# Create a custom dataset for testing using ImageFolder and apply the class_to_label mapping for test dataset
test_dataset = ImageFolder(root=test_data_dir, transform=preprocess_test_image)

# Replace the class labels in the test dataset with numerical labels using the mapping
modified_test_samples = []
for item in test_dataset.samples:
    path, label = item
    folder_name = os.path.basename(os.path.dirname(path))
    numerical_label = class_to_label.get(folder_name, -1)
    if numerical_label != -1:
        modified_test_samples.append((path, numerical_label))

# Update the test dataset samples with the modified list for testing dataset
test_dataset.samples = modified_test_samples

# Create DataLoader for test dataset
test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

# Define the model architecture
eff_net = EfficientNet.from_pretrained(model_name, num_classes=5)

num_features = eff_net._fc.in_features

eff_net._fc = nn.Linear(num_features, 5)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(eff_net._fc.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eff_net.to(device)

accumulation_steps = 4
# Training loop
for epoch in range(num_epochs):
    eff_net.train()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = eff_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check GPU memory usage and print information
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

        total_samples += labels.size(0)
        total_correct += (torch.argmax(outputs, 1) == labels).sum().item()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Free up memory explicitly
            del inputs, labels, outputs
            gc.collect()
            torch.cuda.empty_cache()
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = eff_net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # Calculate training accuracy
#         _, predicted = torch.max(outputs.data, 1)
#         total_samples += labels.size(0)
#         total_correct += (predicted == labels).sum().item()

    train_accuracy = total_correct / total_samples * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.2f}%')

# Evaluation
eff_net.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = eff_net(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
torch.save(eff_net, 'trained_model_final.pth')
