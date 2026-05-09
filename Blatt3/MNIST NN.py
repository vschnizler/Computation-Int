
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

transform = transforms.ToTensor()

train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
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
    
    model.eval()
    
    run_error = []
    
    optimizer.zero_grad()
    
    for image, label in (train):
        prediction = model(image)
        loss = lossfn(prediction, label.long())
        
        predicted_classes = prediction.argmax(1)
        wrong_count = (predicted_classes != label).type(torch.float).mean().item()
        run_error.append(wrong_count)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return np.average(run_error)
    
def training_loop(training_data, test_data, model, lossfn, learning_rate, epochs):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    error_hist = []
    test_err_hist = []
    
    initial_test_err = np.inf
    
    best_test_err = np.inf
        
    for epoch in range(epochs):
        
        current_run_err = training_run(training_data, model, lossfn, optimizer)
        error_hist.append(current_run_err)
        
        current_test_err = (test_run(test_data, model, lossfn))
        test_err_hist.append(current_test_err)
        
        
        if(current_test_err > initial_test_err):
            
            print("Test Error increase")
           
        if(current_test_err < best_test_err):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_run_err,
            }, "checkpoint_err=", np.round(current_test_err, 6) , ".pth")
            
            best_test_err = current_test_err
        
        initial_test_err = current_test_err
        
        print("Epoch: ", epoch, " / ", epochs, "   Training Error = ", current_run_err, "   Testing Error = ", current_test_err)
    return np.array(error_hist), np.array(test_err_hist)

def test_run(test_data, model, lossfn):
    error = []
    total_wrong = 0
    total_samples = len(test_data)
    with torch.no_grad():
        for image, label in test_data:
            prediction = model(image)
            predicted_classes = prediction.argmax(dim=1)
            wrong_in_batch = (predicted_classes != label).sum().item()
            
            total_wrong += wrong_in_batch
            total_samples += label.size(0)
        
    return total_wrong / total_samples

model = NeuralNetwork().to(device)

checkpoint = torch.load("checkpoint579Test2.pth", weights_only=False)

# model.load_state_dict(checkpoint['model_state_dict'])

error_y, test_error_y = training_loop(train_loader, test_loader, model, lossfn, 0.001, 580)

error_x = np.arange(len(error_y))

plt.plot(error_y, error_x, color="r", label="Training Error")
plt.plot(test_error_y, error_x, color="b", label="Testerror")
plt.grid()
plt.legend(loc="upper right")
plt.show()


