import torch 
import torch.nn as nn                
import torch.nn.functional as F     
import torch.optim as optim          
import matplotlib.pyplot as plt      
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import re
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
def _get_model_versions():
    files = os.listdir(MODELS_DIR)
    versions = []
    for f in files:
        match = re.match(r"model(\d+)\.pt$", f)
        if match:
            versions.append(int(match.group(1)))
    return sorted(versions)
def export_model(model):
    versions = _get_model_versions()
    next_version = (versions[-1] + 1) if versions else 1
    path = os.path.join(MODELS_DIR, f"model{next_version}.pt")
    torch.save(model.state_dict(), path)
    print(f"[✔] Model exported to: {path}")
def import_model(model_class, version=None):
    versions = _get_model_versions()
    if not versions:
        raise FileNotFoundError("No saved models found in the 'models/' directory.")
    if version is None:
        print(f"Available versions: {versions}")
        user_input = input("Enter model version to load or press Enter for latest: ")
        version = int(user_input) if user_input.strip().isdigit() else versions[-1]

    if version not in versions:
        raise ValueError(f"Model version {version} not found. Available: {versions}")
    path = os.path.join(MODELS_DIR, f"model{version}.pt")
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model version {version} loaded from: {path}")
    return model
def model_io(choice, model_class=None, model_instance=None):
    if choice == 1:
        if model_instance is None:
            print("You must pass a model_instance to export.")
        else:
            export_model(model_instance)

    elif choice == 2:
        if model_class is None:
            print("You must pass a model_class to import.")
        else:
            return import_model(model_class)
    else:
        print("Invalid option. Choose 'import' or 'export'.")
df = pd.read_csv("data.csv")
x_df, y_df = df.iloc[:, :5], df.iloc[:, 5:]
x = torch.tensor(x_df.values, dtype=torch.float32)
y = torch.tensor(y_df.values, dtype=torch.float32)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 100)  
        self.fc2 = nn.Linear(100, 50)  
        self.fc3 = nn.Linear(50, 5)
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)          
        return x
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
def train(x, y, epochs):
    print("Train Started.")
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()  
        pred = model(x)            
        loss = criterion(pred, y) 
        loss.backward()             
        optimizer.step()
        if epoch % 500 == 0:
            mae = torch.mean(torch.abs(pred - y)).item()
            print(f"[Epoch] : {epoch}/{epochs} | [Loss] : {loss.item():.4f} | [MAE] : {mae:.4f}")
def evaluate(x, y, num_examples=10):
    print("Evaluation Started.")
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)
        mae = torch.mean(torch.abs(pred - y)).item()

        print(f"[Loss] : {loss.item():.4f} | [MAE] : {mae:.4f}")
        print("\n--- Example Predictions ---")
        for i in range(min(num_examples, len(pred))):
            print("---------------------")
            print("Input     :", [f"{v:.2f}" for v in x[i].tolist()])
            print("Predicted :", [f"{v:.2f}" for v in pred[i].tolist()])
            print("Target    :", [f"{v:.2f}" for v in y[i].tolist()], "\n")
def predict(x):
    model.eval()
    with torch.no_grad():
        pred = model(x)
    return pred.tolist()
def main():
    global model 
    choice = input("Train a new model or load an existing one? (T/L): ").strip().lower()
    if choice == "t":
        epochs = input("How many epochs? ")
        train(x, y, int(epochs))
        evaluate(x, y, 20)
        save_choice = input("Do you want to export this trained model? (y/n): ").strip().lower()
        if save_choice == "y":
            model_io(1 ,model_instance=model)
    elif choice == "l":
        loaded_model = model_io(2, model_class=Net)
        if loaded_model:
            model = loaded_model
        evaluate(x, y, 10)
    else:
        print("Invalid choice. Please type 'T' or 'L'.")
    while True:
        play = []
        for i in range(5):
            while True:
                try:
                    playnum = int(input(f"{i + 1} NUM: "))
                    break
                except ValueError:
                    print("Please enter a valid number.")
            play.append(torch.tensor(playnum, dtype=torch.float32))
        play  = torch.tensor(play, dtype=torch.float32)
        pred = predict(play)
        for i in range(len(pred)):
            print(f"{i + 6} NUM: {pred[i]:.2f}")
        print("\n")
main()