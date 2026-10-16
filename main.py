import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import pathlib as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.models(x)
        
df = pd.read_csv(pl.Path("data/student_performance.csv"))
X_df = df[["study_hours", "absences", "previous_grade"]]
Y_df = df[["final_grade"]]

X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(
    X_df, Y_df, test_size =0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.float32)

X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test = torch.tensor(Y_test_df.values, dtype=torch.float32)

model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

loss_history = []
for i in range(0,1000):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, Y_train)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    """ if (i+1) %100 == 0:
        print(f"Epoch [{i+1}/5000], Loss: {loss.item():.4f}") """

plt.figure(figsize=(8,5))
plt.plot(loss_history, color='purple', linewidth=2)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, Y_test)

    print(f"\n--- Evaluation on Unseen Data ---")
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    
    print("\nSample Predictions vs Real Values:")
    for i in range(5):
        print(f"Predicted: {test_pred[i].item():.2f} | Real: {Y_test[i].item():.2f}")