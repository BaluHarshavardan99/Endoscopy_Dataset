import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the folder containing images
folder = "/home/easgrad/baluhars/PIPELINE/FINAL_TEST/KEY_FRAMES/5"


# Get the list of image files in the folder
image_files = [os.path.join(folder, file) for file in os.listdir(folder)]
index_list = []
for image_path in image_files:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["an endoscopic image from endoscopy camera", "a presentation slide on endoscopy techniques"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # Find the highest probability in the list of probs
    max_prob = max(probs[0])

    # Find the index of the highest probability
    index = list(probs[0]).index(max_prob)
    if index==1:
        print("Index is 1 and image is ......", image_path)
    index_list.append(index)


print("Index list is .......................", index_list)
