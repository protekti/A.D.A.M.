import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from os import walk
import matplotlib.pyplot as plt

# Load datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Set device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device.")

# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# Testing function
def test(dataloader, model, loss_fn, oldloss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct  # Calculate accuracy
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n")
    
    # Determine if training should continue
    done = not (0 < oldloss - test_loss < 0.6)
    
    return test_loss, accuracy, done  # Return both loss and accuracy

# Initialize model, loss function, and optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Training loop
done = True
t = 0
old_loss = 0

# Initialize metrics
loss_differences = []
test_losses = []
accuracies = []
plt.ion()
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

while done:
    print(f"Old loss: {old_loss}")
    print(f"Epoch {t + 1}\n---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    
    # Capture both test_loss and accuracy
    test_loss, accuracy, done = test(test_dataloader, model, loss_fn, old_loss)
    
    # Store metrics
    loss_difference = old_loss - test_loss
    loss_differences.append(loss_difference)
    test_losses.append(test_loss)
    accuracies.append(accuracy)  # Use the captured accuracy
    
    old_loss = test_loss
    t += 1

    # Update plots
    axs[0].cla()
    axs[0].plot(loss_differences, label='Old Loss - Test Loss', color='blue')
    axs[0].set_title('Loss Difference')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss Difference')
    axs[0].legend()

    axs[1].cla()
    axs[1].plot(test_losses, label='Test Loss', color='orange')
    axs[1].set_title('Test Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    axs[2].cla()
    axs[2].plot(accuracies, label='Accuracy', color='green')
    axs[2].set_title('Accuracy')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Accuracy (%)')
    axs[2].legend()

    plt.pause(0.1)

plt.ioff()
plt.show()

# Save the model
n = 0
for (dirpath, dirnames, filenames) in walk("./models/"):
    n += 1
    break

torch.save(model.state_dict(), f"./models/model{n + 1}.pth")
print(f"Saved PyTorch Model State to model{n + 1}.pth")
