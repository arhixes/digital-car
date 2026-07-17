import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src import config
from src.agent.core_agent import get_embeddings

def init_vector_store():
    """
    Инициализирует локальное хранилище Qdrant и создает коллекцию для мануала.
    """
    print(f"Инициализация локальной базы Qdrant по пути: {config.QDRANT_PATH}")
    
    # Подключаем локальный Qdrant
    client = QdrantClient(path=config.QDRANT_PATH)
    collection_name = "elantra_manual"
    
    # Получаем модель эмбеддингов
    embeddings = get_embeddings()
    vector_size = len(embeddings.embed_query("тест"))
    print(f"Размерность векторов модели GigaChat: {vector_size}")

    # Проверяем, существует ли коллекция
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        print(f"Создаем новую коллекцию: {collection_name}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print("Коллекция успешно создана!")
    else:
        print(f"Коллекция {collection_name} уже существует, пересоздание не требуется.")
        
    return client

if __name__ == "__main__":
    init_vector_store()