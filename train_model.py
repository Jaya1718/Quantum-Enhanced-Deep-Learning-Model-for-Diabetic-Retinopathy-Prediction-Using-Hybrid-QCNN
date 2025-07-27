import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from quantums_model import HybridQCNN

# Load preprocessed data
df = pd.read_csv("train_quantum_ready.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].astype(int).values

print("Label Distribution:\n", pd.Series(y).value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()

# Model setup
model = HybridQCNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Training loop
print("🚀 Training HybridQCNN with real health features...")
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        y_pred_train = torch.argmax(output, axis=1)
        acc = accuracy_score(y_train, y_pred_train)
    print(f"Epoch {epoch+1}/30 - Loss: {loss.item():.4f} - Train Acc: {acc*100:.2f}%")

# Evaluation
model.eval()
y_pred = torch.argmax(model(X_test), axis=1).detach().numpy()
acc = accuracy_score(y_test, y_pred)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")

# Save model
torch.save(model.state_dict(), "hybrid_qcnn_model.pth")
print("💾 Model saved as hybrid_qcnn_model.pth")
