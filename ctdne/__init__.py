from typing import List, Tuple, Optional

import numpy as np
import torch
from temporal_walk import TemporalWalk

from ctdne.dynamic_embedding_model import DynamicEmbeddingModel


class NodeNotFoundError(Exception):
    def __init__(self, node_id: int):
        self.message = f'{node_id} is not a found in the graph'

class EmbeddingModel:

    def __init__(
            self,
            num_walks: int,
            len_walk: int,
            random_picker_type: str,
            d_embed: int,
            max_node_count: int,
            max_time_capacity: Optional[int]=None,
            batch_size: int = 10_000,
            embedding_epochs: int=5,
            context_len: int=5,
            learning_rate: float=0.001,
            device:Optional[torch.device]=None,
            min_count: int=1,
            workers: int=6
    ):
        self.num_walks = num_walks
        self.len_walk = len_walk
        self.random_picker_type = random_picker_type
        self.fill_value = 0
        self.batch_size = batch_size
        self.embedding_epochs = embedding_epochs

        self.temporal_walk = TemporalWalk(num_walks, len_walk, random_picker_type, max_time_capacity)
        self.embedding_model = DynamicEmbeddingModel(
            max_node_count=max_node_count, embedding_dim=d_embed,
            context_size=context_len, lr=learning_rate, device=device
        )

    def add_temporal_edges(self, edges: List[Tuple[int, int, int]]):
        self.temporal_walk.add_multiple_edges(edges)

    def get_walks_from_network(self, node_ids=None):
        if node_ids is None:
            nodes = self.temporal_walk.get_node_ids()
        else:
            nodes = node_ids

        walks_for_nodes = self.temporal_walk.get_random_walks_for_nodes("Random", nodes, self.fill_value)
        flattened_walks = self._flatten_walks(walks_for_nodes)
        return flattened_walks

    def train_embedding_with_walks(self, walks):
        self._train_model_with_walks(walks)

    def train_embedding_with_current_state(self):
        walks = self.get_walks_from_network()
        self.train_embedding_with_walks(walks)

    @staticmethod
    def _flatten_walks(walks_for_nodes):
        flattened_walks = []
        for walks in walks_for_nodes.values():
            for walk in walks:
                flattened_walks.append(np.trim_zeros(walk, 'b'))
        return flattened_walks

    def _train_model_with_walks(self, walks):
        self.embedding_model.train_model(walks, batch_size=self.batch_size, epochs=self.embedding_epochs)

    def get_embedding_for_node(self, node):
        try:
            return self.embedding_model.get_embedding(node)
        except KeyError:
            raise NodeNotFoundError(node)

    def get_embedding_for_all_nodes(self):
        return self.embedding_model.get_all_embeddings()
