import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


# تنظیم بذر تصادفی برای PyTorch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

# تنظیم بذر تصادفی برای NumPy
np.random.seed(0)

# تنظیم بذر تصادفی برای Python
random.seed(0)

# Function to load images from directory
def load_images_from_directory(directory):
    images = []
    labels = []
    label = 1 if str(directory).split('\\')[1]=='Cancer' else 0
    for folder_name in os.listdir(directory)[:]:  # Use all available images
        image_path = os.path.join(directory, folder_name)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(label)
    return images, labels

# Load training and testing images and labels
def load_data():
    train_cancer_images, train_cancer_labels = load_images_from_directory("skin_data\\Cancer\\Training")
    train_non_cancer_images, train_non_cancer_labels = load_images_from_directory("skin_data\\Non_Cancer\\Training")
    test_cancer_images, test_cancer_labels = load_images_from_directory("skin_data\\Cancer\\Testing")
    test_non_cancer_images, test_non_cancer_labels = load_images_from_directory("skin_data\\Non_Cancer\\Testing")

    train_images = np.array(train_cancer_images + train_non_cancer_images)
    train_labels = np.array(train_cancer_labels + train_non_cancer_labels)
    test_images = np.array(test_cancer_images + test_non_cancer_images)
    test_labels = np.array(test_cancer_labels + test_non_cancer_labels)

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    train_images_tensor = torch.tensor(train_images).permute(0, 3, 1, 2)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_images_tensor = torch.tensor(test_images).permute(0, 3, 1, 2)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    return train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

# Load data
train_images, train_labels, test_images, test_labels = load_data()

# Create DataLoader for training data
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Change output size to 2 for binary classification

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), "skin_cancer_model_resnet.pth")
print("Model saved")


# def calculate_accuracy(model, data_loader):
#     correct = 0
#     total = 0
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / total
#     return accuracy

# # Calculate accuracy on training data
# train_accuracy = calculate_accuracy(model, train_loader)
# print(f"Training Accuracy: {train_accuracy:.2f}")


# # Predictions on test set
# model.eval()
# with torch.no_grad():
#     outputs = model(test_images)
#     _, predicted_labels = torch.max(outputs, 1)

# # Convert tensors to numpy arrays
# predicted_labels = predicted_labels.numpy()
# true_labels = test_labels.numpy()

# # 2. Confusion Matrix
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
# print("Confusion Matrix:")
# print(conf_matrix)

# # 3. Precision and Recall
# print("Classification Report:")
# print(classification_report(true_labels, predicted_labels))

# # 4. F1-score
# f1_score = classification_report(true_labels, predicted_labels, output_dict=True)['weighted avg']['f1-score']
# print(f"F1-Score: {f1_score:.2f}")