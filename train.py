import torch
from torch import optim
from torch import nn
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
    model.cuda()
    lr = 3e-4
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epochs = 30
    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []

    print("Training...")
    for epoch in range(epochs):

        model.train()
        corrects = 0
        train_num = 0
        for step, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28, 28)
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            preds = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss.item()
            corrects += torch.sum(preds == labels.data)
            train_num += images.size(0)

        train_losses.append(loss / len(train_loader))
        train_acces.append(corrects.double().item() / train_num)
        print(f"Train: {epoch+1:>03d}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acces[-1]:.4f}")

        model.eval()
        corrects = 0
        test_num = 0
        for step, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28, 28)
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            preds = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            loss += loss.item()
            corrects += torch.sum(preds == labels.data)
            test_num += images.size(0)

        test_losses.append(loss / len(test_loader))
        test_acces.append(corrects.double().item() / test_num)
        print(f"Test : {epoch+1:>03d}: Test  Loss: {test_losses[-1]:.4f}, Test  Acc: {test_acces[-1]:.4f}")

    torch.save({"state_dict": model.state_dict(),}, "./checkpoint/rnn.pth")
    print('Save model done!')