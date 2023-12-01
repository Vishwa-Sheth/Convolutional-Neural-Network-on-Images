import os
import torch
from torchvision import transforms
from PIL import Image
import PIL
import torch.nn as nn
import torch.nn.functional as F

# Load the pre-trained model
model_filename = '0502-01-668575956Sheth.ZZZ'  # Replace with the actual model file
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
num_classes = 9
model = Net(num_classes)  # Define your model architecture here
model.load_state_dict(torch.load(model_filename))
model.eval()
labels_list = ["Circle", "Square", "Octagon", "Heptagon", "Nonagon", "Star", "Hexagon", "Pentagon", "Triangle"]
labels_list.sort()

# Data transformations (the same as during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
])

# List all image files in the current directory
current_directory = os.getcwd()
image_files = [f for f in os.listdir(current_directory) if f.endswith('.png')]

# Perform inference
results = {}
for image_file in image_files:
    try:
        image_path = os.path.join(current_directory, image_file)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension (batch size of 1)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        predicted_class = labels_list[predicted.item()]  # Use the class labels list from your training code

        results[image_file] = predicted_class
    except (OSError, PIL.UnidentifiedImageError):
        print(f"Skipped {image_file} due to an error")

# Output the results
for image_file, predicted_class in results.items():
    print(f"{image_file}: {predicted_class}")
