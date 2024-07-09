import os
from PIL import Image
from torchvision import transforms
import torch
import timm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(image_path, transform):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if transform:
            image = transform(image)
        return image
    except (OSError, IOError) as e:
        print(f"Error opening image {image_path}: {e}")
        return None

# Load the saved model weights
model = timm.create_model('adv_inception_v3', pretrained=True, num_classes=2)
model = model.to(device)
model.load_state_dict(torch.load('/home/easgrad/baluhars/1234567890/ENDO_CLASSIFIER/model_weights.pth'))
model.eval()  # Set the model to evaluation mode

# Define the data transformation
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_file_location = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/image_paths.csv"
df = pd.read_csv(csv_file_location)

# Create a new column for classification labels
df['classification'] = ''

for index, row in df.iterrows():
    image_path = row['image_path']
    try:
        # Load and preprocess the image
        image = load_image(image_path, transform)
        if image is not None:
            image_tensor = image.unsqueeze(0).to(device)  # Move the tensor to the appropriate device

            # Classify the image
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output.data, 1)

            # Save the classification label
            if predicted.item() == 0:
                label = 'endoscopy'
            else:
                label = 'non-endoscopy'
            df.at[index, 'classification'] = label
    except Exception as e:
        print(f'Error classifying {image_path}: {e}')

# Save the updated CSV file
df.to_csv('/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/classification_csv/output_csv_file.csv', index=False)