import json
import random
def remove_non_numeric(obj):
    if isinstance(obj, dict):
        return {key: remove_non_numeric(value) for key, value in obj.items() if key in ["label", "edges", "features"]}
    elif isinstance(obj, list):
        return [remove_non_numeric(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, str):
        return ''.join(char for char in obj if char.isdigit())
    else:
        return obj

# Process 10 different JSON files
for i in range(10):
    with open(f'synthetic_{i}.json', 'r') as f:
        json_data = json.load(f)
        cleaned_json = remove_non_numeric(json_data)

        # Write cleaned JSON to a new file
        with open(f'cleaned_synthetic_{i}.json', 'w') as outfile:
            json.dump(cleaned_json, outfile, indent=2)
