import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.optim import Adam
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random
import numpy as np
from torch_geometric.nn import Node2Vec

G = nx.Graph()

G.add_node('A', data=8)
G.add_node('B', data=6)
G.add_node('C', data=4)
G.add_node('D', data=7)
G.add_node('E', data=5)
G.add_node('F', data=10)

edges = [
    ('A', 'B', 40),
    ('A', 'C', 60),
    ('A', 'D', 50),
    ('A', 'E', 70),
    ('A', 'F', 80),
    ('B', 'C', 30),
    ('B', 'D', 20),
    ('B', 'E', 40),
    ('B', 'F', 50),
    ('C', 'D', 40),
    ('C', 'E', 20),
    ('C', 'F', 30),
    ('D', 'E', 30),
    ('D', 'F', 20),
    ('E', 'F', 10),
]

for u, v, weight in edges:
    G.add_edge(u, v, weight=weight)

def build_pyg_data(graph, path):
    node_indices = {node: i for i, node in enumerate(graph.nodes)}
    edge_index = []
    edge_attr = []
    for u, v, data in graph.edges(data=True):
        edge_index.append([node_indices[u], node_indices[v]])
        edge_attr.append(data['weight'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.tensor([[graph.nodes[node]['data']] for node in graph.nodes], dtype=torch.float)
    path_indices = [node_indices[node] for node in path]
    path_tensor = torch.tensor(path_indices, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, path=path_tensor)

optimal_path = ['A', 'B', 'C', 'E', 'F', 'D', 'A']
data = build_pyg_data(G, optimal_path)

def generate_node2vec_embeddings(data, embedding_dim=32, walk_length=10, context_size=8, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True):
    node2vec = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size,
                        walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q, sparse=sparse)
    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    def train_node2vec():
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, 101):
        loss = train_node2vec()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    return node2vec.embedding.weight.data


class Node2VecModel(nn.Module):
    def __init__(self, embedding_dim):
        super(Node2VecModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.path_fc = nn.Linear(embedding_dim, 6)

    def forward(self, x, path):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        energy_output = F.softplus(self.fc3(out)).squeeze()

        path_features = x[path]
        path_logits = self.path_fc(x)
        path_probs = F.softmax(path_logits, dim=1)

        path_indices = [0]
        remaining_nodes = list(range(1, len(x)))
        for _ in range(len(x) - 1):
            prob_dist = path_probs[path_indices[-1]].detach().numpy()
            next_node = remaining_nodes.pop(torch.multinomial(torch.tensor(prob_dist[remaining_nodes]), 1).item())
            path_indices.append(next_node)
        path_indices.append(0)
        path_indices = torch.tensor(path_indices, dtype=torch.long)

        return energy_output, path_indices

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

embedding_dim = 32
node_embeddings = generate_node2vec_embeddings(data, embedding_dim=embedding_dim)

model = Node2VecModel(embedding_dim=embedding_dim)

def loss_function(data, energy_output, path_indices, initial_energy=450, charge_capacity=450, penalty_weight=0.5,
                  reg_weight=0.1):
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    total_distance_energy = 0
    current_energy = initial_energy
    charged = False

    last_node = path_indices[0].item()
    for i in range(1, len(path_indices)):
        next_node = path_indices[i].item()

        mask = (edge_index[0] == last_node) & (edge_index[1] == next_node)
        if mask.sum() == 0:
            mask = (edge_index[1] == last_node) & (edge_index[0] == next_node)

        if mask.sum() != 1:
            print(f"Warning: Edge between {last_node} and {next_node} not found or multiple edges found")
            continue

        edge_distance = edge_attr[mask].squeeze().item()
        total_distance_energy += edge_distance
        current_energy -= edge_distance

        if not charged and next_node == 2:
            current_energy = charge_capacity
            charged = True

        last_node = next_node

    total_remaining_energy = max(0, current_energy)
    node_energy_loss = torch.sum((energy_output - data.x.squeeze()) ** 2)
    energy_allocation_penalty = torch.sum(torch.relu(8 - energy_output))
    reg_loss = reg_weight * sum(torch.sum(param ** 2) for param in model.parameters())

    total_loss = (total_distance_energy / 1000 +
                  node_energy_loss / 1000 +
                  total_remaining_energy / 1000 +
                  penalty_weight * energy_allocation_penalty / 10 +
                  reg_loss)

    return total_loss


initial_lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

warmup_epochs = 50
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda
    epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1)

num_epochs = 500
best_loss = float('inf')
best_path = None
best_energy_output = None
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    energy_output, path_indices = model(node_embeddings, data.path)
    loss = loss_function(data, energy_output, path_indices)
    loss.backward()
    optimizer.step()

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    losses.append(loss.item())

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_path = path_indices.clone()
        best_energy_output = energy_output.clone()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Optimal Path:", [int(node) for node in best_path])
print("Optimal Energy Allocation per Node:", best_energy_output.detach().numpy())

plt.plot(range(num_epochs), losses)
plt.xlabel('Training Episode')
plt.ylabel('Loss')
plt.xlim(left=0, right=num_epochs)
plt.ylim(bottom=0)
plt.show()
