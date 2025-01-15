import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Define a transform to normalize the data and apply data augmentation
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

# Define the model with dropout and batch normalization
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.log_softmax(self.fc3(x))
        return x

# Initialize the model, define the loss function, optimizer, and scheduler
model = DigitRecognizer()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if __name__ == "__main__":
    # Training loop with validation and early stopping
    epochs = 20
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        val_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in valloader:
                output = model(images)
                val_loss += criterion(output, labels).item()

                # Calculate accuracy
                preds = torch.exp(output).argmax(dim=1)
                accuracy += (preds == labels).sum().item()

        val_loss /= len(valloader)
        accuracy /= len(val_subset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(trainloader):.3f} - Validation loss: {val_loss:.3f} - Validation Accuracy: {accuracy:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), "../models/best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Save the final model
    torch.save(model.state_dict(), "../models/final_model.pth")
