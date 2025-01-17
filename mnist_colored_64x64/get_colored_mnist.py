
import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt

def get_mnist_colored_trainloader(size=(64, 64), batch_size=8):
    # Define transform pipeline
    transform = Compose([
        ToTensor(),
        Resize(size),  # Resize to 64x64
        #Lambda(lambda x: torch.stack([x, x, x], dim=0)),  # Grayscale to RGB
        Lambda(lambda x: x * torch.rand(3, 1, 1))  # Add random RGB scaling
    ])

    # Load MNIST dataset
    mnist = MNIST(root='data', train=True, download=True, transform=transform)
    trainloader = DataLoader(mnist, batch_size, shuffle=True)
    return trainloader

# Example usage
trainloader = get_mnist_colored_trainloader()
trainiter = iter(trainloader)
images, labels = next(trainiter) 

_, axs = plt.subplots(1, 8, figsize=(12, 12))
axs = axs.flatten()

for img, ax in zip(images.cpu(), axs):
    ax.imshow(img.permute(1, 2, 0))

plt.show()
