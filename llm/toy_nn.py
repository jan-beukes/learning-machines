import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LOAD_MODEL = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

def main():
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = NeuralNetwork(2, 2)
    model = model.to(device)

    if not LOAD_MODEL:
        train_model(model, train_loader)
    else:
        try:
            state = torch.load("model.pt")
            model.load_state_dict(state)
            print(f"Loaded pretrained model from 'model.pt'")
        except:
            train_model(model, train_loader)

    print(compute_accuracy(model, test_loader))
    torch.save(model.state_dict(), "model.pt")

def train_model(model, train_loader):
    # Stochastic Gradient decent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            logits = model(features) # output values
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}/{num_epochs}"
                  f"  | Batch {batch_idx+1}/{len(train_loader)}"
                  f"  | Train Loss: {loss:.2f}")
    return model

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        # Don't do gradient calculations for back prop since we are done training
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = predictions == labels
        correct += torch.sum(compare).item() # give float from the 0d tensor
        total_examples += len(compare)
    return correct / total_examples

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2nd hidden layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                # Output layer
                torch.nn.Linear(20, num_outputs),
                )

    def forward(self, x):
        logits = self.layers(x)
        return logits
if __name__ == "__main__": main()
