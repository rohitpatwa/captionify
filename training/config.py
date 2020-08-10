import torch

root_folder = "../flickr8k_train_sample/images"
annotation_file = "../flickr8k_train_sample/captions.txt"
testdir = "../test_samples"
num_workers=2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
train_CNN = True
save_checkpoint_path="../trained_models"
load_checkpoint_path="../trained_models/checkpoint_90.pth.tar"
save_model=True

embed_size = 256
hidden_size = 256
num_layers = 3
learning_rate = 3e-4
num_epochs = 100