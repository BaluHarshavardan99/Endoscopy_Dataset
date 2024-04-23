import torch
import clip
from PIL import Image
import os
from utils import COLOR

class CLIP_CLF:
    def __init__(self, model_name):
        # super(clip_clf, self).__init__()
        self.model, preprocess = clip.load("ViT-B/32", device=self.device)
    
# device = "cuda" if torch.cuda.is_available() else "cpu"
    def forward(self):
        image_files = [os.path.join(self.args.frame_path, file) for file in os.listdir(self.args.frame_path)]
        index_list = []
        for image_path in image_files:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text = clip.tokenize(["an endoscopic image from endoscopy camera", "a presentation slide on endoscopy techniques"]).to(self.device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            # Find the highest probability in the list of probs
            max_prob = max(probs[0])

            index = list(probs[0]).index(max_prob)
            if index==1:
                print(Color.GREEN + "Index is 1 and image is"+ image_path + Color.END)
            index_list.append(index)

        print(Color.GREEN + "Index list is"+ index_list + Color.END)
        # print("### Index list is .......................", index_list)
