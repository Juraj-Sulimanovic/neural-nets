import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

x_0, y_0 = train_set[0]

trans = transforms.Compose([transforms.ToTensor()])

x_0_tensor = trans(x_0)
x_0_tensor.to(device).device

x_0_gpu = x_0_tensor.to(device)

train_set.transform = trans
valid_set.transform = trans

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

n_classes = 10
input_size = 1 * 28 * 28
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512), # Input
    nn.ReLU(),                  # Activation for input
    nn.Linear(512, 512),        # Hidden
    nn.ReLU(),                  # Activation for hidden
    nn.Linear(512, n_classes)   # Output
]

model = nn.Sequential(*layers)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

epochs = 5
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()


### Prediction
prediction = model(x_0_gpu)
prediction.argmax(dim=1, keepdim=True)
print(y_0)

### Input image
plt.imshow(x_0, cmap="gray")
plt.axis("off")
plt.show()
