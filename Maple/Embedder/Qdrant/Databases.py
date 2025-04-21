from Maple.Embedder.Qdrant.QdrantBase import QdrantBase


class MS1FullCollection(QdrantBase):
    def __init__(self):
        super().__init__(
            collection_name="ms1_full_collection",
            memory_strategy="memory",
            label_alias="peak_id",
            embedding_dim=128,
            memmap_threshold=None,
            delete_existing=False,
        )


class MS2Reference(QdrantBase):
    def __init__(self):
        super().__init__(
            collection_name="ms2_chemotype_reference",
            memory_strategy="disk",
            label_alias="chemotype",
            embedding_dim=128,
            memmap_threshold=20000,
            delete_existing=False,
        )
