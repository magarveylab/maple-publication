import pickle
from glob import glob

from tqdm import tqdm

from Maple.Embedder.Qdrant.QdrantBase import QdrantBase, batchify


class MS1FullCollection(QdrantBase):
    def __init__(
        self, use_cloud_service: bool = True, delete_existing: int = False
    ):
        super().__init__(
            collection_name="ms1_full_collection",
            memory_strategy="disk",
            label_alias="peak_id",
            embedding_dim=128,
            memmap_threshold=None,
            delete_existing=delete_existing,
            use_cloud_service=use_cloud_service,
        )

    def initial_upload(self, embedding_dir: str):
        filenames = glob(f"{embedding_dir}/*.pkl")
        for fp in tqdm(filenames, desc="Uploading MS1 embeddings"):
            peaks = pickle.load(open(fp, "rb"))
            batches = batchify(peaks, bs=1000)
            for batch in tqdm(batches):
                ids = []
                vectors = []
                payloads = []
                for peak in batch:
                    peak_id = peak["ms1_peak_id"]
                    embedding = peak["embedding"]
                    ids.append(peak_id)
                    vectors.append(embedding)
                    payloads.append(
                        {
                            "mass": peak["mass"],
                            "rt": peak["rt"],
                            "intensity": peak["intensity"],
                            "adduct_type": peak["adduct_type"],
                            "mzml_id": peak["mzml_id"],
                            "strain_id": peak["strain_id"],
                        }
                    )
                self.upload_data_batch(
                    ids=ids, vectors=vectors, payloads=payloads
                )
        self.index_collection()


class MS2Reference(QdrantBase):
    def __init__(self, use_cloud_service: bool = True):
        super().__init__(
            collection_name="ms2_chemotype_reference",
            memory_strategy="disk",
            label_alias="chemotype",
            embedding_dim=128,
            memmap_threshold=20000,
            delete_existing=False,
            use_cloud_service=use_cloud_service,
        )
