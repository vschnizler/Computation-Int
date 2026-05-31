import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

# Must be set before any torch.load calls
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

lossfn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class PreprocessedMNIST(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = torch.load(images_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)
        print(f"Loaded dataset from {images_path}. Total samples: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class RawPlusSobel:
    def __init__(self):
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1,-2,-1],
                                 [ 0, 0, 0],
                                 [ 1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel = torch.cat([sobel_x, sobel_y], dim=0)  # (2, 1, 3, 3)

    def __call__(self, x):
        # x: (1, 28, 28)
        with torch.no_grad():
            out  = func.conv2d(x.unsqueeze(0), self.kernel, padding=1)  # (1, 2, 28, 28)
            gx, gy = out[0, 0], out[0, 1]                                # (28, 28)
            edge = torch.sqrt(gx**2 + gy**2).unsqueeze(0)               # (1, 28, 28)
        return torch.cat([x, edge], dim=0)   # (2, 28, 28)


PREPROCESSED_TRAIN_IMAGES_PATH = './data/mnist_train_images_sobel.pt'
PREPROCESSED_TRAIN_LABELS_PATH = './data/mnist_train_labels_sobel.pt'
PREPROCESSED_TEST_IMAGES_PATH  = './data/mnist_test_images_sobel.pt'
PREPROCESSED_TEST_LABELS_PATH  = './data/mnist_test_labels_sobel.pt'

if all(os.path.exists(p) for p in [
    PREPROCESSED_TRAIN_IMAGES_PATH, PREPROCESSED_TRAIN_LABELS_PATH,
    PREPROCESSED_TEST_IMAGES_PATH,  PREPROCESSED_TEST_LABELS_PATH,
]):
    print("Pre-processed data found. Skipping reprocessing.")
else:
    print("Pre-processing MNIST with RawPlusSobel...")
    base_transform   = transforms.ToTensor()
    sobel_transform  = RawPlusSobel()
    raw_train = datasets.MNIST(root='./data', train=True,  download=True, transform=base_transform)
    raw_test  = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    def process(dataset):
        imgs, lbls = [], []
        for image, label in dataset:
            imgs.append(sobel_transform(image))
            lbls.append(label)
        return torch.stack(imgs), torch.tensor(lbls)

    os.makedirs('./data', exist_ok=True)
    train_imgs, train_lbls = process(raw_train)
    test_imgs,  test_lbls  = process(raw_test)

    torch.save(train_imgs, PREPROCESSED_TRAIN_IMAGES_PATH)
    torch.save(train_lbls, PREPROCESSED_TRAIN_LABELS_PATH)
    torch.save(test_imgs,  PREPROCESSED_TEST_IMAGES_PATH)
    torch.save(test_lbls,  PREPROCESSED_TEST_LABELS_PATH)
    print("Pre-processing complete.")

train_dataset = PreprocessedMNIST(PREPROCESSED_TRAIN_IMAGES_PATH, PREPROCESSED_TRAIN_LABELS_PATH)
test_dataset  = PreprocessedMNIST(PREPROCESSED_TEST_IMAGES_PATH,  PREPROCESSED_TEST_LABELS_PATH)

num_workers = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32,  shuffle=False, num_workers=num_workers, pin_memory=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # Input: (B, 2, 28, 28)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),   # → (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (B, 64,  7,  7)

            nn.Flatten(),                                  # → (B, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.stack(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def training_run(train, model, lossfn, optimizer):
    model.train()
    run_error = []
    optimizer.zero_grad()

    for image, label in train:
        image, label = image.to(device), label.to(device)

        # Noise on raw channel only; sobel channel stays clean
        noisy_image = image.clone()
        noisy_image[:, 0, :, :] += torch.randn_like(image[:, 0, :, :]) * 0.1

        prediction = model(noisy_image)
        loss = lossfn(prediction, label.long())

        predicted_classes = prediction.argmax(1)
        wrong_count = (predicted_classes != label).type(torch.float).mean().item()
        run_error.append(wrong_count)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.average(run_error)


def test_run(test_data, model, lossfn):
    model.eval()   # disables Dropout during evaluation
    total_wrong, total_samples = 0, 0

    with torch.no_grad():
        for image, label in test_data:
            image, label = image.to(device), label.to(device)
            predicted_classes = model(image).argmax(dim=1)
            total_wrong   += (predicted_classes != label).sum().item()
            total_samples += label.size(0)

    return total_wrong / total_samples


def training_loop(training_data, test_data, model, lossfn, learning_rate, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    error_hist, test_err_hist = [], []
    print("In Training Loop")

    for epoch in range(epochs):
        current_run_err  = training_run(training_data, model, lossfn, optimizer)
        current_test_err = test_run(test_data, model, lossfn)

        error_hist.append(current_run_err)
        test_err_hist.append(current_test_err)

        print(f"Epoch {epoch+1}/{epochs} — train err: {current_run_err:.4f}, test err: {current_test_err:.4f}")

        if current_test_err < 0.006:
            fname = f"checkpoint_err={np.round(current_test_err, 6)}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_run_err,
            }, fname)
            print("New Best Model saved:", fname)
            return np.array(error_hist), np.array(test_err_hist)

    return np.array(error_hist), np.array(test_err_hist)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
model = NeuralNetwork().to(device)

error_y, test_error_y = training_loop(train_loader, test_loader, model, lossfn, 0.001, 40)

error_x = np.arange(len(error_y))
plt.plot(error_x, error_y,      color="r", label="Training Error")
plt.plot(error_x, test_error_y, color="b", label="Test Error")
plt.grid()
plt.legend(loc="upper right")
plt.show()