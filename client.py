import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import flwr as fl
from collections import OrderedDict
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(client_id, num_clients=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    num_samples = len(trainset) // num_clients
    start_idx = client_id * num_samples
    end_idx = start_idx + num_samples
    
    client_trainset = Subset(trainset, range(start_idx, end_idx))
    
    return client_trainset, testset

def train_with_dp(model, trainloader, epochs, device, epsilon=1.0, delta=1e-5):
    model = ModuleValidator.fix(model)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    privacy_engine = PrivacyEngine()
    
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=1.0, 
    )
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        epsilon_spent = privacy_engine.get_epsilon(delta)
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(trainloader):.4f}, ε={epsilon_spent:.2f}")
    
    return epsilon_spent

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_clients=5, epsilon=1.0, delta=1e-5):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN()
        self.epsilon = epsilon
        self.delta = delta

        trainset, testset = load_data(client_id, num_clients)
        self.trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=1000, shuffle=False)
        
        print(f"Client {client_id}: {len(trainset)} training samples")
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        epochs = config.get("epochs", 1)
        epsilon_spent = train_with_dp(
            self.model, self.trainloader, epochs, 
            self.device, self.epsilon, self.delta
        )
        
        print(f"Client {self.client_id}: Privacy budget spent: ε={epsilon_spent:.2f}")
        
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "epsilon": float(epsilon_spent),
            "client_id": self.client_id
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy, correct, total = test(self.model, self.testloader, self.device)
        
        print(f"Client {self.client_id}: Test Accuracy = {accuracy:.2f}%")
        
        return float(len(self.testloader.dataset)), {
            "accuracy": float(accuracy),
            "correct": int(correct),
            "total": int(total)
        }

def start_client(client_id, num_clients=5, epsilon=1.0):
    client = FlowerClient(client_id, num_clients, epsilon)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client
    )

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    print(f"Starting client {client_id} with ε={epsilon}")
    start_client(client_id, num_clients, epsilon)