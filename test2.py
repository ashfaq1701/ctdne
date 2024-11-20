import os.path
import time
import pickle
import pandas as pd
from ctdne import EmbeddingModel


NUM_WALKS = 100
LEN_WALK = 100
D_EMBED = 128

DATA_DIR = '/Users/ashfaq/Documents/traces/filtered_data_files'

def convert_pd_to_edges(filepath):
    df = pd.read_parquet(filepath)
    edges = []
    for _, row in df.iterrows():
        edges.append((row['u'].astype(int), row['i'].astype(int), row['ts'].astype(int)))
    return edges

if __name__ == '__main__':
    start_time = time.time()

    embedding_model = EmbeddingModel(
        num_walks=NUM_WALKS,
        len_walk=LEN_WALK,
        random_picker_type="Linear",
        d_embed=D_EMBED,
        max_node_count=70_000
    )

    for i in range(3):
        filepath = os.path.join(DATA_DIR, f'data_{i}.parquet')
        edges = convert_pd_to_edges(filepath)
        embedding_model.add_temporal_edges(edges)

    walks = embedding_model.get_walks_from_network()
    print(f"Max walk len: {max([len(walk) for walk in walks])}")
    embedding_model.train_embedding_with_walks(walks)
    embeddings_for_all_nodes = embedding_model.get_embedding_for_all_nodes()

    pickle.dump(embeddings_for_all_nodes, open('data/embeddings_for_all_nodes_2.pkl', 'wb'))
    print(f'Total runtime: {time.time() - start_time:.2f} seconds')
