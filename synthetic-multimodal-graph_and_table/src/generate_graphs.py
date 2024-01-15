import json
import random

num_graphs = 20
nodes_per_graph = 5

data_list = []

for graph_id in range(num_graphs):
    edges = []
    for i in range(nodes_per_graph):
        for j in range(i + 1, nodes_per_graph):
            if random.random() < 0.5:
                edges.append([i, j])

    features = {}
    for node_id in range(nodes_per_graph):
        feature_values = [random.uniform(0.0, 1.0) for _ in range(5)]
        features[str(node_id)] = feature_values

    label = random.randint(0, 1)

    graph_data = {
        "label": label,
        "edges": edges,
        "features": features
    }

    with open(f"{graph_id}.json", "w") as json_file:
        json.dump(graph_data, json_file, indent=2)

    data_list.append(graph_data)

print(f"Generated {num_graphs} synthetic graphs in .json format.")
