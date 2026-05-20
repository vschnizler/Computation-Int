import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as func
from torch.utils.data import random_split, TensorDataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def save_results(model, error_y, test_error_y, label):
    # extract layer shapes
    shapes = []
    for layer in model.stack:
        if isinstance(layer, nn.Linear):
            shapes.append(f"{layer.in_features}")
    shapes.append("1")  # final output
    
    shape_str = "-".join(shapes)
    filename = f"{label}_{shape_str}.npz"
    
    np.savez(filename, 
             train_error=error_y, 
             test_error=test_error_y,
             shapes=np.array(shapes))
    
    print(f"Saved to {filename}")

def gen_class_A():
    u = np.arange(1, 250)
    x = -3.0 + 0.1 * np.sin(0.08 * u) * (u + 15)
    y = 2 + 0.1 * np.cos(0.08 * u) * (u + 15)
    labels = np.ones(len(u))
    return np.column_stack([x, y, labels])

def gen_class_B():
    u = np.arange(1, 250)
    x = -3.0 + 0.1 * np.sin(0.08 * u + 1.6) * (u + 15)
    y = 2 + 0.1 * np.cos(0.08 * u + 1.6) * (u + 15)
    labels = np.zeros(len(u))
    return np.column_stack([x, y, labels])

def transform_to_polar(x1, x2):
    r = np.sqrt(np.abs(x1**2 + x2**2))
    phi = np.arctan2(x2, x1)
    
    return r, phi

A = gen_class_A()
B = gen_class_B()

polar_A = np.column_stack((*transform_to_polar(A[:, 0], A[:, 1]), np.ones(len(A))))
polar_B = np.column_stack((*transform_to_polar(B[:, 0], B[:, 1]), np.zeros(len(B))))

print(polar_A)

spiral_A = torch.Tensor(A)
spiral_B = torch.Tensor(B)

spiral_A_polar = torch.Tensor(polar_A)
spiral_B_polar = torch.Tensor(polar_B)

data = torch.cat([spiral_A, spiral_B], dim=0)  

data_polar = torch.cat([spiral_A_polar, spiral_B_polar], dim=0)

x = data[:, :2]   
y = data[:, 2]
dataset = TensorDataset(x, y)

train_set, test_set = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

x_polar = data_polar[:, :2]   
y_polar = data_polar[:, 2]
dataset_polar = TensorDataset(x_polar, y_polar)

train_set_polar, test_set_polar = random_split(dataset_polar, [0.8, 0.2])

train_loader_polar = DataLoader(train_set_polar, batch_size=32, shuffle=True)
test_loader_polar = DataLoader(test_set_polar, batch_size=32, shuffle=True)

lossfn = nn.BCELoss()

def plot_heatmap(x_range, y_range, model):
    xx, yy = np.meshgrid(x_range, y_range)
    grid = torch.Tensor(np.column_stack([xx.ravel(), yy.ravel()]))
    
    with torch.no_grad():
        z = model(grid).squeeze().numpy().reshape(xx.shape)
    
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, z, levels=50, cmap='RdBu')
    plt.colorbar(label='Model Output')
    plt.grid()
    plt.title('Model Output Heatmap')
    plt.show()

def plot_heatmap_polar(r_range, phi_range, model):
    rr, pp = np.meshgrid(r_range, phi_range)
    grid = torch.Tensor(np.column_stack([rr.ravel(), pp.ravel()]))
    
    with torch.no_grad():
        z = model(grid).squeeze().numpy().reshape(rr.shape)
    
    plt.figure(figsize=(8, 8))
    plt.contourf(rr, pp, z, levels=50, cmap='RdBu')
    plt.colorbar(label='Model Output')
    plt.xlabel('r')
    plt.ylabel('phi')
    plt.grid()
    plt.title('Model Output Heatmap (Polar)')
    plt.show()

class SpiralClassifier(nn.Module):
    def __init__(self):
        super(SpiralClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.stack = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
    )

    def forward(self, x):
        
        x = self.flatten(x)
        return self.stack(x)
    
def training_run(train, model, lossfn, optimizer):
        
    model.train()
    
    run_error = []
    
    optimizer.zero_grad()
    
    for datapoint, label in (train):
        
        noisy_datapoint = datapoint + torch.randn_like(datapoint) * 0.06    
        
        prediction = model(noisy_datapoint)
        loss = lossfn(prediction, label.float().unsqueeze(1))
        
        predicted_classes = (prediction >= 0.5).float().squeeze(1)
        wrong_count = (predicted_classes != label).type(torch.float).mean().item()
        run_error.append(wrong_count)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return np.average(run_error)

def training_loop(training_data, test_data, model, lossfn, learning_rate, epochs, label: str):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
            
            print("Test Error increase", current_test_err, " epoch =", epoch)
        
        if(current_test_err < best_test_err):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_run_err,
            }, "BestModelspiral" + label + ".pth")
            
            print("Test Error decrease")
            best_test_err = current_test_err
            if(best_test_err < 0.01):
                return np.array(error_hist), np.array(test_err_hist)
        
        initial_test_err = current_test_err
        
    return np.array(error_hist), np.array(test_err_hist)

def test_run(test_data, model, lossfn):
    total_wrong = 0
    total_samples = 0  # start at 0
    with torch.no_grad():
        
        for datapoint, label in test_data:
            
            prediction = model(datapoint)
            predicted_classes = (prediction >= 0.5).float().squeeze(1)  # threshold instead of argmax
            total_wrong += (predicted_classes != label).sum().item()
            total_samples += label.size(0)
            
    return total_wrong / total_samples

# checkpoint = torch.load("checkpoint_err=0.020072.pth", weights_only=False)

# model.load_state_dict(checkpoint['model_state_dict'])

# plt.figure(figsize=(8, 8))
# plt.scatter(spiral_A[:, 0], spiral_A[:, 1], c='red', label='Class A', s=10)
# plt.scatter(spiral_B[:, 0], spiral_B[:, 1], c='blue', label='Class B', s=10)
# plt.grid()
# plt.legend()
# plt.title('Spiral Dataset')
# plt.show()

# model_cartesian = SpiralClassifier().to(device)

# x1 = np.linspace(-30.0, 30.0, int(60/0.25))
# x2 = np.linspace(-30.0, 30.0, int(60/0.25))

# error_y, test_error_y = training_loop(train_loader, test_loader, model_cartesian, lossfn, 0.001, 1000, "Cartesian")

# save_results(model_cartesian, error_y, test_error_y, "Cartesian")

# error_x = np.arange(len(error_y))

# plt.plot(error_x, error_y, color="r", label="Training Error")
# plt.plot(error_x, test_error_y, color="b", label="Testerror")
# plt.grid()
# plt.legend(loc="upper right")
# plt.show()

# plot_heatmap(x1, x2, model_cartesian)

model_polar = SpiralClassifier().to(device)

error_y, test_error_y = training_loop(train_loader_polar, test_loader_polar, model_polar, lossfn, 0.0001, 1000, "Polar")

save_results(model_polar, error_y, test_error_y, "Polar")

error_x = np.arange(len(error_y))

plt.plot(error_x, error_y, color="r", label="Training Error")
plt.plot(error_x, test_error_y, color="b", label="Testerror")
plt.grid()
plt.legend(loc="upper right")
plt.show()

r = np.linspace(0, 30.0, 200)
phi = np.linspace(-np.pi, np.pi, 200)

plot_heatmap_polar(r, phi, model_polar)


