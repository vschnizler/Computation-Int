
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)

lossfn = nn.CrossEntropyLoss()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.stack = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
    )

    def forward(self, x):
        
        x = self.flatten(x)
        return self.stack(x)

def training_run(train, model, lossfn, optimizer):
    
    run_error = []
    
    optimizer.zero_grad()
    
    for image, lable in (train):
        prediction = model(image)
        loss = lossfn(prediction, lable.long())
        
        run_error.append(float(loss.detach()))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return np.average(run_error)
    
def training_loop(training_data, test_data, model, lossfn, learning_rate, epochs):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    error_hist = []
    test_err_hist = []
    
    for epoch in range(epochs):
        error_hist.append(training_run(training_data, model, lossfn, optimizer))
        test_err_hist.append(test_run(test_data, model, lossfn))
        
    return np.array(error_hist), np.array(test_err_hist)

def test_run(test_data, model, lossfn):
    error = []
    
    with torch.no_grad():
        for image, label in test_data:
            prediction = model(image)
            loss = lossfn(prediction, label.long())
            
            error.append(float(loss))
        
    return np.average(error)

model = NeuralNetwork().to(device)

error_y, test_error_y = training_loop(train_loader, test_loader, model, lossfn, 0.001, 500)

error_x = np.arange(len(error_y))

plt.plot(error_y, error_x, color="r", label="Training Error")
plt.plot(test_error_y, error_x, color="b", label="Testerror")
plt.show()


