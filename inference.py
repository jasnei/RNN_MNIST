import matplotlib.pyplot as plt
import torch
from data import get_dataset
from model_design import RNNClasssifier

if __name__ == '__main__':

    train_loader, test_loader = get_dataset()
    print(f"train_loader.len: {len(train_loader)}")
    print(f"test_loader.len: {len(test_loader)}")

    model = RNNClasssifier(
            input_dim=28,
            hidden_dim=128,
            layer_dim=1,
            output_dim=10
        )

    checkpoint = torch.load("./checkpoint/rnn.pth")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Load model done")

    for step, (images, labels) in enumerate(test_loader):
        if step > 0:
            break
    images = images.view(-1, 28, 28)
    output = model(images)
    preds = torch.argmax(output, dim=1)
    coorects = torch.sum(preds == labels)
    acc = coorects / labels.size(0)
    print(f"Acc: {acc:.4f}")

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(10, 10, i + 1)
        img = images[i, :]
        img = img.data.numpy().reshape(28, 28)
        plt.imshow(img, 'gray')
        plt.title(preds[i].item())
        plt.axis('off')
        plt.subplots_adjust(hspace=0.5, wspace=0.1)
    plt.show()