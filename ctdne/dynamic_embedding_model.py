import torch
import torch.nn as nn
import torch.optim as optim


class DynamicEmbeddingModel(nn.Module):
    def __init__(self, max_node_count, embedding_dim, lr=0.001, max_norm=1.0, alpha=0.5, context_size=5, device=None):
        """
        Computes Dynamic Embedding

        Args:
            max_node_count (int): Maximum number of nodes.
            embedding_dim (int): Dimensionality of node embeddings.
            lr (float): Learning rate for the optimizer.
            max_norm (float): Maximum norm for node embeddings.
            alpha (float): Decay factor for discounting context weights.
            context_size (int): Size of the context window for training.
        """
        super(DynamicEmbeddingModel, self).__init__()
        self.max_node_count = max_node_count
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.alpha = alpha
        self.context_size = context_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Internal mapping for node IDs to indices
        self.node_id_to_idx = {}
        self.current_idx = 0

        # Fixed embedding matrix
        self.embeddings = nn.Embedding(max_node_count, embedding_dim, max_norm=max_norm).to(self.device)
        nn.init.uniform_(self.embeddings.weight, -1, 1)

        # Optimizer for the embeddings
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _get_or_create_index(self, node_id):
        """
        Get or create the internal index for a node ID.

        Args:
            node_id (int): The external node ID.

        Returns:
            int: Internal index corresponding to the node ID.
        """
        if node_id not in self.node_id_to_idx:
            if self.current_idx >= self.max_node_count:
                raise ValueError(f"Exceeded maximum node count ({self.max_node_count}).")
            # Assign a new index
            self.node_id_to_idx[node_id] = self.current_idx
            self.current_idx += 1
        return self.node_id_to_idx[node_id]

    def forward(self, node_ids):
        """
        Retrieve embeddings for the given node IDs.

        Args:
            node_ids (list or tensor): List or tensor of node IDs.

        Returns:
            torch.Tensor: Embeddings for the provided node IDs.
        """
        indices = torch.tensor(
            [self._get_or_create_index(node_id) for node_id in node_ids], dtype=torch.long, device=self.device
        )
        return self.embeddings(indices)

    def train_model(self, walks, batch_size=128, epochs=10):
        """
        Train the node embeddings based on walks in batches.

        Args:
            walks (list): List of walks (each walk is a list of node IDs).
            batch_size (int): Size of the training batches.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            total_loss = 0
            num_batches = len(walks) // batch_size

            for batch_idx in range(num_batches):
                batch_walks = walks[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_loss = self._train_on_walk_batch(batch_walks)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                total_loss += batch_loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

    def _train_on_walk_batch(self, batch_walks):
        """
        Train the model on a batch of walks.

        Args:
            batch_walks (list): A batch of walks (list of node ID lists).

        Returns:
            torch.Tensor: The loss for the batch of walks.
        """
        batch_loss = 0
        for walk in batch_walks:
            walk_loss = self._train_on_walk(walk)
            batch_loss += walk_loss
        return batch_loss

    def _train_on_walk(self, walk):
        total_loss = 0
        walk = torch.tensor(
            [self._get_or_create_index(node_id) for node_id in walk],
            dtype=torch.long,
            device=self.device  # Move tensor to the correct device
        )

        for i, current_node_idx in enumerate(walk):
            start = max(0, i - self.context_size)
            end = min(len(walk), i + self.context_size + 1)

            context = torch.cat([walk[start:i], walk[i + 1:end]]).to(self.device)
            distances = torch.tensor(
                [abs(i - j) for j in range(start, end) if j != i],
                dtype=torch.float32,
                device=self.device
            )
            discounts = self.discount_function(distances)

            current_embedding = self.embeddings(current_node_idx).unsqueeze(0)
            context_embeddings = self.embeddings(context)

            similarities = torch.matmul(context_embeddings, current_embedding.T).squeeze()
            context_loss = torch.sum(discounts * (1 - similarities))

            norm = torch.norm(current_embedding)
            context_loss += torch.relu(norm - self.max_norm)

            total_loss += context_loss

        return total_loss

    def discount_function(self, distances):
        """
        Apply exponential decay to context weights based on distance.

        Args:
            distances (torch.Tensor): Tensor of distances.

        Returns:
            torch.Tensor: Discounted weights based on distances.
        """
        return torch.exp(-self.alpha * distances)

    def get_embedding(self, node_id):
        """
        Retrieve the embedding for a specific node.

        Args:
            node_id (int): External node ID.

        Returns:
            torch.Tensor: Embedding for the specified node.
        """
        idx = self.node_id_to_idx.get(node_id)
        if idx is None:
            raise KeyError(f"Node {node_id} is not in the embeddings.")
        return self.embeddings(torch.tensor([idx], dtype=torch.long, device=self.device)).detach().cpu()

    def get_all_embeddings(self):
        """
        Retrieve all node embeddings as a dictionary.

        Returns:
            dict: A dictionary with node IDs as keys and embeddings as values.
        """
        return {node_id: self.embeddings(torch.tensor([idx], dtype=torch.long, device=self.device)).detach().cpu()
                for node_id, idx in self.node_id_to_idx.items()}
