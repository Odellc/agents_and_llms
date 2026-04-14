import litellm
from litellm import completion
import os

if os.getenv("OPENAI_API_KEY"):
    litellm.openapi_key = os.getenv("OPENAI_API_KEY")

if (litellm.openapi_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected VOC API key, using VOC API endpoint")


# Imports and global config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math, time, random
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Hint: Use transforms.ToTensor() and transforms.Normalize((0.5,), (0.5,))
train_tfms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_tfms  = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load datasets and create dataloaders

train_ds = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=train_tfms
)
test_ds = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=test_tfms
)

# Create a small validation split from the training set
val_ratio = 0.1
val_size = int(len(train_ds) * val_ratio)
train_size = len(train_ds) - val_size
train_ds, val_ds = torch.utils.data.random_split(
    train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(1234)
)

# Create Data loaders: train_loader, val_loader, test_loader

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # shuffle for training
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)   # no shuffle for validation
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)  # no shuffle for test


len(train_ds), len(val_ds), len(test_ds)


# Show some sample images
def show_images(dataloader):
    batch = next(iter(dataloader))
    images, labels = batch
    fig, ax = plt.subplots(3, 5, figsize=(5, 3))
    for i in range(3):
        for j in range(5):
            ax[i, j].imshow(images[i * 5 + j].squeeze(), cmap="gray")
            ax[i, j].set_title(f"Label: {labels[i * 5 + j].item()}")
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.show()


show_images(train_loader)


# Tiny CNN model definition

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = TinyCNN().to(device)

# Show the model size
sum(p.numel() for p in model.parameters()), model.__class__.__name__


@torch.no_grad()
def accuracy_from_logits(logits, y):
    """Compute accuracy from model logits and true labels.

    logits: (B, C), y: (B,)
    """

    # Use argmax to get predicted class. Hint: dim = 1
    preds = logits.argmax(dim=1)

    # Get boolean tensor of correct predictions, hint: use ==
    correct = preds == y

    # Return mean accuracy of the batch
    return correct.float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()  # zero the parameter gradients
        logits = model(x)  # run the model on the inputs to get the logits
        loss = criterion(logits, y)  # calculate the loss using criterion
        loss.backward()  # run backpropagation to compute gradients
        optimizer.step()  # run a single optimization step


        # stats
        batch = x.size(0)
        running_loss += loss.item() * batch
        running_acc += accuracy_from_logits(logits, y) * batch
        n += batch
    return running_loss / n, running_acc / n

# Validation/test loop
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        batch = x.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy_from_logits(logits, y) * batch
        n += batch
    return total_loss / n, total_acc / n


# Function to create a fresh model instance for each optimizer
def make_fresh_model():
    m = TinyCNN().to(device)
    return m

# We use CrossEntropyLoss for classification problems
criterion = nn.CrossEntropyLoss()

# Function to create an optimizer for a given model
def make_optimizer(name, params):
    if name == "sgd":
        return optim.SGD(params, **optim_cfgs["sgd"])
    if name == "adam":
        return optim.Adam(params, **optim_cfgs["adam"])
    if name == "rmsprop":
        return optim.RMSprop(params, **optim_cfgs["rmsprop"])
    raise ValueError("Unknown optimizer")


# Optimizer configurations
optim_cfgs = {
    'sgd': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
    'adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 1e-4},   
    'rmsprop': {'lr': 0.001, 'alpha': 0.99, 'weight_decay': 1e-4},
}

# Train each optimizer for a few epochs and store histories
EPOCHS = 5  # keep short for live runtime
histories = {}

for opt_name in ["sgd", "adam", "rmsprop"]:
    print(f"\n=== Training with {opt_name.upper()} ===")
    set_seed(1234)  # reset for fair comparison
    model = make_fresh_model()
    optimizer = make_optimizer(opt_name, model.parameters())

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        print(
            f"Epoch {epoch:02d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}"
        )

    histories[opt_name] = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    }

print("\nDone training all optimizers.")

# Plot validation accuracy curves
plt.figure(figsize=(7, 4))
for name, h in histories.items():
    plt.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Optimizer Bake-off: Val Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# Evaluate the best optimizer on the test set
best_name = max(histories.keys(), key=lambda n: max(histories[n]["val_acc"]))
print("Best by val acc:", best_name)

set_seed(1234)
best_model = make_fresh_model()
best_opt = make_optimizer(best_name, best_model.parameters())
for _ in range(3):
    train_one_epoch(best_model, train_loader, criterion, best_opt, device)
_, test_acc = evaluate(best_model, test_loader, criterion, device)
print(f"Test accuracy with {best_name.upper()}: {test_acc:.4f}")