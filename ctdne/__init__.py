from typing import List, Tuple, Optional

from temporal_walk import TemporalWalk
from gensim.models import Word2Vec
import numpy as np

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
            max_time_capacity: Optional[int]=None,
            embedding_epochs: int=5,
            window: int=10,
            min_count: int=1,
            workers: int=6
    ):
        self.num_walks = num_walks
        self.len_walk = len_walk
        self.random_picker_type = random_picker_type
        self.fill_value = 0

        self.temporal_walk = TemporalWalk(num_walks, len_walk, random_picker_type, max_time_capacity)
        self.embedding_model = Word2Vec(
            vector_size=d_embed, epochs=embedding_epochs, window=window, min_count=min_count, sg=1, workers=workers
        )

    def add_temporal_edges(self, edges: List[Tuple[int, int, int]]):
        self.temporal_walk.add_multiple_edges(edges)

    def get_walks_from_network(self):
        nodes = self.temporal_walk.get_node_ids()
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
        return [np.trim_zeros(walk, 'b') for walks in walks_for_nodes.values() for walk in walks]

    def _train_model_with_walks(self, walks):
        if not self.embedding_model.wv.key_to_index:
            self.embedding_model.build_vocab([[str(node) for node in walk] for walk in walks])
        else:
            self.embedding_model.build_vocab([[str(node) for node in walk] for walk in walks], update=True)

        string_walks = [[str(node) for node in walk] for walk in walks]
        self.embedding_model.train(string_walks, total_examples=len(walks), epochs=self.embedding_model.epochs)

    def get_embedding_for_node(self, node):
        try:
            return self.embedding_model.wv[str(node)]
        except KeyError:
            raise NodeNotFoundError(node)

    def get_embedding_for_all_nodes(self):
        return {int(node): self.embedding_model.wv[node] for node in self.embedding_model.wv.index_to_key}
