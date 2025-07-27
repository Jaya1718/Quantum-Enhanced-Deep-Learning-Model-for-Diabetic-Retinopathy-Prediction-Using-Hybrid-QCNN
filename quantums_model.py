import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_qubits))

    def forward(self, x):
        return torch.stack([
            torch.tensor(quantum_circuit(xi, self.q_weights), dtype=torch.float32)
            for xi in x
        ])

class HybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = QuantumLayer()
        self.fc1 = nn.Linear(n_qubits, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.q_layer(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
