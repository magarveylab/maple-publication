import argparse
import os
from glob import glob

import requests
from tqdm import tqdm

from Maple.Embedder import curdir
from Maple.Embedder.Qdrant.Databases import MS1FullCollection


def restore_ms1_qdrant(embedding_dir: str):
    db = MS1FullCollection(delete_existing=True)
    db.initial_upload(embedding_dir=embedding_dir)


def restore_other_qdrant():
    node_url = "http://localhost:6333"
    local_snapshot_paths = glob(f"{curdir}/QdrantSnapshots/*.snapshot")
    for snapshot_path in tqdm(local_snapshot_paths):
        snapshot_name = os.path.basename(snapshot_path)
        collection_name = snapshot_name.split("-")[0]
        requests.post(
            f"{node_url}/collections/{collection_name}/snapshots/upload?priority=snapshot",
            files={"snapshot": (snapshot_name, open(snapshot_path, "rb"))},
        )


parser = argparse.ArgumentParser(description="Building Qdrant Databases")

parser.add_argument(
    "-ms1_embedding_dir",
    help="Download MS1 embeddings from Zenodo",
    default=f"ms1_embeddings",
)

if __name__ == "__main__":
    args = parser.parse_args()
    ms1_embedding_dir = args.ms1_embedding_dir
    restore_ms1_qdrant(embedding_dir=ms1_embedding_dir)
    restore_other_qdrant()
