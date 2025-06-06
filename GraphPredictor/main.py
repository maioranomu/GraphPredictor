import torch 
import torch.nn as nn                
import torch.nn.functional as F     
import torch.optim as optim          
import matplotlib.pyplot as plt      
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
            print(f"[Epoch] : {epoch} | [Loss] : {loss.item():.4f} | [MAE] : {mae:.4f}")

def evaluate(x, y, num_examples=5):
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
            print(f"Input     : {x[i].tolist()}")
            print(f"Predicted : {pred[i].tolist()}")
            print(f"Target    : {y[i].tolist()}\n")

train(x, y, 20000)
evaluate(x, y)