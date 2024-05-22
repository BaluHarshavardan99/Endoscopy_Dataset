import os
from PIL import Image
from torchvision import transforms
import torch
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_images_from_folder(folder_path, transform=None):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if transform:
                    image = transform(image)
                images.append(image)
            except (OSError, IOError) as e:
                print(f"Error opening image {image_path}: {e}")
    return images

# Load the saved model weights
model = timm.create_model('adv_inception_v3', pretrained=True, num_classes=2)
model = model.to(device)
model.load_state_dict(torch.load('model_weights.pth'))

# Define the data transformation
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images from the test folder
test_folder = '/home/easgrad/baluhars/1234567890/ENDO_CLASSIFIER/test_images'
test_images = load_images_from_folder(test_folder, transform)

# Move the images to the GPU
test_images = [image.to(device) for image in test_images]

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for image in test_images:
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        total += 1
        correct += (predicted == 1).sum().item()  # Assuming label 0 is 'endoscopy'
        print(predicted)

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
