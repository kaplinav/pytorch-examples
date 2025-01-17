
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_cifar_testloader(size=(64, 64), batch_size=8, subset=1000):
    # Create a DataLoader with a given batch size
    #dataloader = DataLoader(Subset(dataset, range(256)), batch_size, shuffle=True)
    #dataloader = DataLoader(dataset, batch_size, shuffle=True)
    transform = transforms.Compose([
        # Ensure images are 64x64
        transforms.Resize(size),
        # Convert images to PyTorch tensors.
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )

    if subset != None:
        trainset = Subset(trainset, range(subset))

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )

    return trainloader


import matplotlib.pyplot as plt
import numpy as np


# functions to show an image.
def imshow(img):
    # Unnormalize.
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


cifarloader = get_cifar_testloader()

# Get some random training images.
dataiter = iter(cifarloader)
images, labels = next(dataiter)

# Show images.
imshow(torchvision.utils.make_grid(images))
