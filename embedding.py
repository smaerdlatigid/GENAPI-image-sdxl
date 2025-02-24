import time
import torch
from PIL import Image
import open_clip

class ImageTextEmbedding:
    def __init__(self, model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b', device='mps'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, image_path):
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, texts):
        text_inputs = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_inputs)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def get_label_probabilities(self, image_path, labels):
        t1 = time.time()
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(labels)

        label_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("Time taken:", time.time() - t1)
        return label_probs

# Example usage:
if __name__ == "__main__":
    embedding_model = ImageTextEmbedding()
    labels = ["a mushroom", "a dog", "a cat"]
    image_path = "images/test_1.webp"
    
    label_probs = embedding_model.get_label_probabilities(image_path, labels)
    print("Label probs:", label_probs)  # prints: [[1., 0., 0.]]