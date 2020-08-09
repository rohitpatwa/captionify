import config
import utils
from model import CNNtoRNN
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle

def generate_caption(img_path):
    # Load vocab
    vocab = pickle.load(open('../flickr8k_vocab.pkl', 'rb'))

    # Create transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Read and transform input image
    input_img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

    # Create model instance
    model = CNNtoRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(config.device)

    # Load checkpoint
    checkpoint = torch.load(config.load_checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])

    # Generate output
    caption = model.caption_image(input_img.to(config.device), vocab)[1:-1]
    caption = " ".join(caption)
    print("OUTPUT: " + caption)
    return caption

if __name__=="__main__":
    img_path = '../test_samples/dog.jpg'
    generate_caption(img_path)