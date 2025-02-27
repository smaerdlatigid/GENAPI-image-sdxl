import time
import torch
from PIL import Image
import open_clip
import requests
from functools import lru_cache

class ImageTextEmbedding:
    def __init__(self, model_name='hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        if model_name.startswith('hf-hub:'):
            self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @lru_cache(maxsize=32)
    def encode_image(self, image_input):
        """Encode image from either a file path or URL"""
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_input, stream=True).raw)
            else:
                image = Image.open(image_input)
        else:
            image = image_input
            
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, texts):
        text_inputs = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_inputs)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def get_label_probabilities(self, image_input, labels):
        t1 = time.time()
        image_features = self.encode_image(image_input)
        text_features = self.encode_text(labels)

        label_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("Time taken:", time.time() - t1)
        return label_probs

# Example usage:
if __name__ == "__main__":
    # Test with Hugging Face model
    embedding_model = ImageTextEmbedding('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    
    # Test with local image
    labels = ["a mushroom", "a dog", "a cat"]
    image_path = "images/test_1.webp"
    label_probs = embedding_model.get_label_probabilities(image_path, labels)
    print("Local image probs:", label_probs)
    
    # Test with online image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    label_probs = embedding_model.get_label_probabilities(url, labels)
    print("Online image probs:", label_probs)