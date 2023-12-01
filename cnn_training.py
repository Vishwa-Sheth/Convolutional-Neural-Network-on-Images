import os
import zipfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Path to the zip file
zip_file_path = os.path.expanduser("~/Desktop/geometry_dataset.zip")

# Directory where you want to extract the contents
extracted_dir = os.path.expanduser('~/Desktop/extracted_data/')

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Create an ImageFolder dataset without sorting
dataset = datasets.ImageFolder(root=extracted_dir, transform=transform)

# Sort the dataset based on filenames
dataset.samples = sorted(dataset.samples, key=lambda x: x[0])

# Lists to store training and testing data
train_data = []
test_data = []
labels_list = ["Circle", "Square", "Octagon", "Heptagon", "Nonagon", "Star", "Hexagon", "Pentagon", "Triangle"]
labels_list.sort()

# Define the split based on the condition i % 10000 < 8000
for i, (image, label) in enumerate(dataset):
    label = torch.tensor((int(i/10000)))

    if i % 10000 < 8000:
        train_data.append((image, label))
    else:
        test_data.append((image, label))

print(len(train_data))
print(len(test_data))

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



# Define your CNN architecture
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

# Create an instance of the network
num_classes = 9  # Number of classes
model = Net(num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    train_losses = []  # To store training loss for plotting
    train_accuracies = []  # To store training accuracy for plotting
    test_losses = []  # To store test loss for plotting
    test_accuracies = []  # To store test accuracy for plotting
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = (total_correct / total_samples) * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accurracy : {train_accuracy:.2f}%')

        # Test the model and calculate test loss and accuracy
        model.eval()
        test_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Calculate accuracy
                predicted = torch.argmax(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = (total_correct / total_samples) * 100

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        model.train()
    
    # Plot training and test loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train')
    plt.plot(range(num_epochs), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train')
    plt.plot(range(num_epochs), test_accuracies, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()



# Train the model
num_epochs = 15
train(model, train_loader, criterion, optimizer, num_epochs)

# Save the trained model
model_filename = '0502-668575956-Sheth.ZZZ'
torch.save(model.state_dict(), model_filename)