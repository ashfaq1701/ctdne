import argparse
import csv
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ctdne import EmbeddingModel


def read_data():
    data_file_path = 'data/data.csv'
    df = pd.read_csv(data_file_path)
    data_np = df[['u', 'i', 't']].to_numpy()
    return [(row[0], row[1], row[2]) for row in data_np]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTDNE test.")

    parser.add_argument('--num_walks', type=int, default=100, help='Number of walks per node.')
    parser.add_argument('--len_walk', type=int, default=200, help='Length of each walk.')
    parser.add_argument('--d_embed', type=int, default=256, help='Dimension of embeddings of each node.')

    args = parser.parse_args()

    start_time = time.time()
    embedding_model = EmbeddingModel(
        num_walks=args.num_walks,
        len_walk=args.len_walk,
        random_picker_type="Linear",
        d_embed=args.d_embed,
    )

    data = read_data()

    edge_addition_start_time = time.time()
    embedding_model.add_temporal_edges(data)
    print(f"Edge addition time: {time.time() - edge_addition_start_time:.2f} seconds")

    walk_extraction_start_time = time.time()
    walks = embedding_model.get_walks_from_network()
    print(f"Walk extraction time: {time.time() - walk_extraction_start_time:.2f} seconds")

    embedding_training_start_time = time.time()
    embedding_model.train_embedding_with_walks(walks)
    print(f"Embedding training time: {time.time() - embedding_training_start_time:.2f} seconds")

    embedding_extraction_start_time = time.time()
    embeddings_for_all_nodes = embedding_model.get_embedding_for_all_nodes()
    print(f"Embedding extraction time: {time.time() - embedding_extraction_start_time:.2f} seconds")

    selected_node = list(embeddings_for_all_nodes.keys())[0]
    print(f"Total nodes: {len(embeddings_for_all_nodes)}, embedding size: {len(embeddings_for_all_nodes[selected_node])}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
