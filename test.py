import csv
import time

from ctdne import EmbeddingModel

NUM_WALKS = 100
LEN_WALK = 50
D_EMBED = 512

def read_data():
    data_file_path = 'data/sample_data.csv'

    data_tuples = []

    with open(data_file_path, mode='r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            u = int(row[0])
            i = int(row[1])
            ts = int(row[2])
            data_tuples.append((u, i, ts))

    return data_tuples

if __name__ == '__main__':
    start_time = time.time()
    embedding_model = EmbeddingModel(
        num_walks=NUM_WALKS,
        len_walk=LEN_WALK,
        random_picker_type="Linear",
        d_embed=D_EMBED
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
