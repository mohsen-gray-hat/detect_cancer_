import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import os

def classify_skin_image(image_path, model_path):
    # Load the pre-trained model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Change output size to 2 for binary classification
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the input image
    image = cv2.imread(image_path)
    original_image = image.copy()  # Keep a copy of the original image

    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image).permute(0, 3, 1, 2)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        if predicted.item() == 1:
            print("Skin cancer detected.")
            # Draw a yellow rectangle around the suspicious area
            cv2.rectangle(original_image, (0, 0), (original_image.shape[1], original_image.shape[0]), (0, 255, 255), 3)
            cv2.putText(original_image, "Skin cancer detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
        else:
            print("No skin cancer detected.")
            cv2.rectangle(original_image, (0, 0), (original_image.shape[1], original_image.shape[0]), (0, 255, 255), 3)
            cv2.putText(original_image, "No skin cancer detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)

    # Display the image with the result
    cv2.imshow("Result", original_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
images=os.listdir(os.getcwd()+'\\'+'images')
for image in images:
    classify_skin_image(f"images\\{image}", "skin_cancer_model_resnet.pth")
    


