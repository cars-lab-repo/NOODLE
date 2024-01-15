import json
import random

# Function to generate random edge
def generate_random_edge():
    edge_names = [
        "top.AES.a1.clk_graphrename_0",
        "top.AES.a1.in_graphrename_1",
        "top.AES.a1.k0_graphrename_2",
        "PartSelect_graphrename_3",
    ]
    return [random.choice(edge_names) for _ in range(3)]

# Function to generate random feature
def generate_random_feature():
    feature_names = [
        "top.AES.a1.clk_graphrename_0",
        "top.AES.a1.in_graphrename_1",
        "top.AES.a1.k0_graphrename_2",
        "PartSelect_graphrename_3",
    ]
    return random.choice(feature_names)

# Function to generate a synthetic JSON
def generate_synthetic_json():
    edges = [generate_random_edge() for _ in range(5)]

    features = {}
    for edge in edges:
        source, target, feature = edge
        if feature:
            features[source] = [[feature, target]]
        else:
            features[source] = [["", target]]

    synthetic_json = {
        "label": random.choice([0, 1]),
        "edges": edges,
        "features": features
    }

    return synthetic_json

# Generate 10 synthetic JSON files
for i in range(10):
    synthetic_json = generate_synthetic_json()
    with open(f'synthetic_{i}.json', 'w') as f:
        json.dump(synthetic_json, f, indent=2)
