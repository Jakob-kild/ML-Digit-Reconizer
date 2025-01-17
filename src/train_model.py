import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training and validation data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_subset, val_subset = random_split(trainset, [train_size, val_size])

trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
valloader = DataLoader(val_subset, batch_size=64, shuffle=True)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x

# Initialize the model, define the loss function, optimizer, and scheduler
model = CNNModel()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if __name__ == "__main__":
    epochs = 50
    best_val_loss = float('inf')
    patience = 7
    counter = 0

    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        val_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in valloader:
                output = model(images)
                val_loss += criterion(output, labels).item()
                preds = output.argmax(dim=1)
                accuracy += (preds == labels).sum().item()

        val_loss /= len(valloader)
        accuracy /= len(val_subset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(trainloader):.3f} - Validation loss: {val_loss:.3f} - Validation Accuracy: {accuracy:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            print(f"Improvement in validation loss. Saving model at epoch {epoch+1}.")
            torch.save(model.state_dict(), "../models/best_model.pth")
        else:
            counter += 1
            print(f"No improvement in validation loss for {counter} epochs.")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Saving final model.")
    torch.save(model.state_dict(), "../models/final_model.pth")
