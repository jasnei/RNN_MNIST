import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

def get_dataset(root="../data/", train_batch_size=64, test_batch_size=100, num_workers=4):

    train_data = MNIST(
        root=root,
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )

    test_data = MNIST(
        root=root,
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader,

if __name__ == '__main__':

    train_loader, test_loader = get_dataset()
    
    print(f"train_loader.len: {len(train_loader)}")
    print(f"test_loader.len: {len(test_loader)}")

    for step, (images, labels)in enumerate(train_loader):
        # print(step)
        # if step > 0:
        break

    img = make_grid(images.reshape(-1, 1, 28, 28))
    img = img.data.numpy().transpose(1, 2, 0)
    plt.figure()
    plt.imshow(img, 'gray')
    plt.show()