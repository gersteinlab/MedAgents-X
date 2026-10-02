import numpy as np
import json
import os
from pymilvus import MilvusClient
import argparse

parent_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", type=str, default="http://localhost:19530")
    parser.add_argument("--base_dir", type=str, default=parent_dir)
    parser.add_argument("--corpus", type=str, default="cpg")
    return parser.parse_args()


args = parse_args()

client = MilvusClient(uri=args.uri)
base_dir = args.base_dir
corpus_name = args.corpus


def data_upload(client: MilvusClient, corpus_name: str):
    global_offset = 0
    batch_size = 10000

    if not client.has_collection(corpus_name):
        client.create_collection(
            collection_name=corpus_name,
            dimension=768,
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
        )

    print(f"data uploading to '{corpus_name}' collection in milvus client...\n")
    chunk_numbers = len(os.listdir(f"{base_dir}/corpus/{corpus_name}/data/"))
    for chunk_num in range(chunk_numbers):
            vectors = np.load(
                f"{base_dir}/corpus/{corpus_name}/vector/{corpus_name}_embeds_{chunk_num}.npy"
            )
            with open(
                f"{base_dir}/corpus/{corpus_name}/json/{corpus_name}_chunk_{chunk_num}.json", 'r'
            ) as jsfile:
                texts = json.load(jsfile)

            data_loaded = [
                {
                    "vector": vectors[ith].tolist(),
                    "text": texts[ith]['contents_chunk'][:30000],
                    "source": corpus_name
                }
                for ith in range(0, len(vectors))
            ]

            for i in range(0, len(data_loaded), batch_size):
                batch = data_loaded[i:i+batch_size]
                for idx, item in enumerate(batch):
                    item['id'] = global_offset + idx
                client.insert(collection_name=corpus_name, data=batch)
                global_offset += len(batch)

    print(f"{corpus_name} Uploaded!")


data_upload(client, corpus_name)
