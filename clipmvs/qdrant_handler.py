import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

class QdrantHandler:
    """
    A class to handle storing and querying embeddings in Qdrant.
    """
    
    idd=0
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the QdrantHandler by reading configuration from a JSON file.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        # Load configuration from the JSON file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.collection_name = config.get("collection_name")
        self.qdrant_client = QdrantClient(url=config.get("url"), api_key=config.get("api_key"))
        
        print(self.qdrant_client.get_collections().collections)
        if self.collection_name not in [x.name for x in self.qdrant_client.get_collections().collections]:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=config.get("vector_size"), distance=Distance.COSINE)
            )
        
    
    def store_embedding(self, embedding, metadata=None):
        """
        Store embeddings in Qdrant.

        Args:
            embedding (numpy.ndarray): The embeddings to store.
            metadata (list): List of metadata dictionaries for each embedding.
        """
        points = [PointStruct(
            id=i+self.idd,
            vector=embedding[i].tolist(),
            payload=metadata[i] if metadata else {}
        ) for i in range(len(embedding))]
        self.idd +=len(embedding) 
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def query_embedding(self, embedding, top_k=10):
        """
        Query embeddings from Qdrant.

        Args:
            embedding (numpy.ndarray): The embedding to query.
            top_k (int): The number of top results to retrieve.

        Returns:
            list: The query results.
        """
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k
        )
        return results

    def close(self):
        """
        Close the Qdrant connection.
        """
        self.qdrant_client.close()
