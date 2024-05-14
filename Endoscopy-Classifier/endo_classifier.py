import os
from PIL import Image
from torchvision import transforms
import torch
import timm
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EndoscopyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Loop through the subdirectories
        for label, dirname in enumerate(['endoscopy', 'non-endoscopy']):
            dir_path = os.path.join(root_dir, dirname)
            for filename in os.listdir(dir_path):
                # Skip hidden files
                if filename.startswith('.'):
                    continue
                file_path = os.path.join(dir_path, filename)
                self.image_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Convert black and white images to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error opening image {image_path}: {e}")
            # Return a placeholder tensor instead of None
            return torch.zeros(3, 456, 456), label
        
        if self.transform:
            # Ensure the image has three channels (RGB format)
            image = image.convert('RGB')
            image = self.transform(image)
        
        return image, label




    # def __getitem__(self, idx):
    #     image_path = self.image_paths[idx]
    #     label = self.labels[idx]
        
    #     try:
    #         image = Image.open(image_path)
    #     except (OSError, IOError) as e:
    #         print(f"Error opening image {image_path}: {e}")
    #         return None, None
        
    #     if self.transform:
    #         image = self.transform(image)
        
    #     return image, label


    # def __getitem__(self, idx):
    #     image_path = self.image_paths[idx]
    #     label = self.labels[idx]
    #     image = Image.open(image_path)
        
    #     if self.transform:
    #         image = self.transform(image)
        
    #     return image, label

# Create data transformations
transform = transforms.Compose([
    transforms.Resize((456, 456)),  # Adjust this based on your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
])

# Create datasets and data loaders
root_dir = '/home/easgrad/baluhars/1234567890/ENDO_CLASSIFIER/dataset'
dataset = EndoscopyDataset(root_dir, transform=transform)



# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Load the EfficientNet-V2 model from timm
model = timm.create_model('adv_inception_v3', pretrained=True, num_classes=2)
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    print("running epoch number .............................. {}" .format(epoch))
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')


# Save the trained model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved successfully")



print("Evaluating the model on the test set ..............................")
# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
