import json

from datasets import load_dataset
from tqdm import tqdm
dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train')

print("Converting dataset to jsonl format")
output_file = "openmathinstruct2.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved as {output_file}")


import json
import os
from tqdm import tqdm

# Path to the saved JSONL file
input_file = "openmathinstruct2.jsonl"

# Output configuration
chunk_size = 250_000
output_dir = "split_openmathinstruct2"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading data from {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in tqdm(f, desc="Reading lines")]

print(f"Splitting data into chunks of {chunk_size} rows")
total = len(data)

for i in range(0, total, chunk_size):
    chunk = data[i:i + chunk_size]
    output_file = os.path.join(output_dir, f"openmathinstruct2_part_{i // chunk_size + 1}.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(chunk, desc=f"Writing part {i // chunk_size + 1}"):
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Split complete. Files saved to: {output_dir}")
