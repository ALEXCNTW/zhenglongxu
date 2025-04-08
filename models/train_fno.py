import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


# FNO
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix, ioj -> boj", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(x.size(0), self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class FNO1dClassifier(nn.Module):
    def __init__(self, modes=8, width=32):
        super().__init__()
        self.fc0 = nn.Linear(3, width)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.conv4 = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = x4 + self.w(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)

#data loading
def load_data():
    data = np.load("data_preprocessing/processed_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    return map(torch.tensor, (X_train, y_train, X_val, y_val))

#training process
def train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_fn(val_out, y_val)
            pred = torch.argmax(val_out, dim=1)
            acc = accuracy_score(y_val.numpy(), pred.numpy())
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")
    torch.save(model.state_dict(), "models/fno_model.pth")
    print("Win-Draw-Loss prediction model training completed, and the model has been saved.")

# start
if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()
    X_train, y_train = X_train.float().unsqueeze(1), y_train.long()
    X_val, y_val = X_val.float().unsqueeze(1), y_val.long()
    model = FNO1dClassifier(modes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val)

