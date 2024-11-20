import torch
import torch.nn as nn
from torch import optim


class DynamicNodeEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, lr=0.001, max_norm=1.0, alpha=0.5, context_size=5):
        """
        Initialize the dynamic node embedding model.

        Args:
            embedding_dim (int): Dimensionality of node embeddings.
            max_norm (float): Maximum norm for node embeddings.
            alpha (float): Decay factor for discounting context weights.
            context_size (int): Size of the context window for training.
        """
        super(DynamicNodeEmbeddingModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.alpha = alpha
        self.context_size = context_size

        # Dictionary to store node embeddings
        self.node_embeddings = nn.ParameterDict()

        # Add a dummy parameter to ensure optimizer initialization
        self.dummy_param = nn.Parameter(torch.zeros(self.embedding_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, node_ids):
        """
        Retrieve or create embeddings for the given node IDs.

        Args:
            node_ids (list): List of node IDs.

        Returns:
            torch.Tensor: Stack of embeddings for the provided node IDs.
        """
        embeddings = []
        for node_id in node_ids:
            if str(node_id) not in self.node_embeddings:
                # Initialize a new embedding for unseen nodes
                new_embedding = nn.Parameter(torch.randn(self.embedding_dim).uniform_(-1, 1))
                self.node_embeddings[str(node_id)] = new_embedding
                self.optimizer.add_param_group({"params": [new_embedding]})
            # Retrieve the embedding
            embeddings.append(self.node_embeddings[str(node_id)])
        return torch.stack(embeddings)

    def constrain_norms(self):
        """
        Ensure embeddings have norms <= max_norm.
        """
        for node_id, embedding in self.node_embeddings.items():
            norm = torch.norm(embedding.data)
            if norm > self.max_norm:
                self.node_embeddings[node_id].data = embedding.data / norm * self.max_norm

    def discount_function(self, distances):
        """
        Apply exponential decay to context weights based on distance.

        Args:
            distances (torch.Tensor): Tensor of distances.

        Returns:
            torch.Tensor: Discounted weights based on distances.
        """
        return torch.exp(-self.alpha * distances)

    def train_model(self, walks, batch_size=128, epochs=10):
        """
        Train the node embeddings based on walks in batches.

        Args:
            walks (list): List of walks (each walk is a list of node IDs).
            batch_size (int): Size of the training batches.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            total_loss = 0
            num_batches = len(walks) // batch_size

            for batch_idx in range(num_batches):
                batch_walks = walks[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_nodes = {str(node) for walk in batch_walks for node in walk}

                batch_loss = 0
                for walk in batch_walks:
                    walk_loss = self._train_on_walk(walk)
                    batch_loss += walk_loss

                for node_id, embedding in self.node_embeddings.items():
                    if node_id not in batch_nodes:
                        embedding.grad = None

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Enforce norm constraints
                self.constrain_norms()

                total_loss += batch_loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

    def _train_on_walk(self, walk):
        """
        Train the model on a single walk.

        Args:
            walk (list): A list of node IDs representing a walk.

        Returns:
            torch.Tensor: The loss for the current walk.
        """
        total_loss = 0

        for i, current_node in enumerate(walk):
            # Extract context window
            start = max(0, i - self.context_size)
            end = min(len(walk), i + self.context_size + 1)

            context = []
            if start < i:
                context.extend(walk[start:i])
            if i + 1 < end:
                context.extend(walk[i + 1:end])

            # Calculate distances for discounting
            distances = torch.tensor(
                [abs(i - j) for j in range(start, end) if j != i], dtype=torch.float32
            )
            discounts = self.discount_function(distances)

            # Get current node embedding
            current_embedding = self([current_node])

            # Calculate context embeddings and update
            context_loss = 0
            for j, context_node in enumerate(context):
                context_embedding = self([context_node])

                # Move embeddings closer
                similarity = torch.dot(
                    current_embedding.squeeze(), context_embedding.squeeze()
                )
                context_loss += discounts[j] * (1 - similarity)

            # Handle self-loops by constraining norm
            norm = torch.norm(current_embedding)
            context_loss += torch.relu(norm - self.max_norm)

            total_loss += context_loss

        return total_loss

    def get_embedding(self, node_id):
        """
        Retrieve the embedding for a specific node.

        Args:
            node_id (int): Node ID.

        Returns:
            torch.Tensor: Embedding for the specified node.
        """
        if str(node_id) not in self.node_embeddings:
            raise KeyError(f"Node {node_id} is not in the embeddings.")
        return self.node_embeddings[str(node_id)].detach().cpu()

    def get_all_embeddings(self):
        """
        Retrieve all node embeddings as a dictionary.

        Returns:
            dict: A dictionary with node IDs as keys and embeddings as values.
        """
        return {node_id: embedding.detach().cpu() for node_id, embedding in self.node_embeddings.items()}
