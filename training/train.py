import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from dataset_loader  import get_loader
from model import CNNtoRNN
import config

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder = config.root_folder,
        annotation_file=config.annotation_file,
        transform=transform,
        num_workers=config.num_workers,
    )

    torch.backends.cudnn.benchmark = True
    device = config.device
    vocab_size = len(dataset.vocab)

    # for tensorboard
    # writer = SummaryWriter("D:\\Datasets\\cv\\image_captioning\\runs\\flickr")
    step = 0

    # initialize model, loss etc

    model = CNNtoRNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.resnet.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = config.train_CNN

    if config.load_model:
        step = load_checkpoint(torch.load(config.load_checkpoint_path), model, optimizer)

    model.train()

    for epoch in range(config.num_epochs):
        # Uncomment the line below to see a couple of test cases

        if epoch%5==0:
            print_examples(model, device, dataset)

        if config.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, f"checkpoint_{epoch}.pth.tar")

        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            # writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()