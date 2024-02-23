import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(64, 128, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(128, 64, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 )
        self.classification_head = nn.Sequential(nn.Linear(64, 20, bias=True),
                                                 nn.ReLU(),
                                                 nn.Linear(20, 10, bias=True), )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(batch_size, -1))


mnist_trainset = datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)
mnist_testset = datasets.MNIST(root="./data", download=True, train=False, transform=ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
batch_size = 50

total_params = 0
for name, param in model.named_parameters():
    params = param.numel()
    total_params += params

EPOCHS = 2
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print(f"Finished Training. Final loss = {loss.item()}, Total params = {total_params}")

checkpoint = {'epoch': EPOCHS, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss.item(), }
print(checkpoint)

torch.save(checkpoint, "checkpoint/checkpoint.pt")