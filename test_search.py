from langchain_qdrant import QdrantVectorStore
from src import config
from src.agent.core_agent import get_embeddings

def test_search():
    embeddings = get_embeddings()
    
    # Подключаемся к нашей свежесозданной базе
    db = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=config.QDRANT_PATH,
        collection_name="elantra_manual"
    )
    
    query = "диагностический разъем тестер"
    
    print(f"\n🔍 Ищем в базе ответ на запрос: '{query}'...")
    docs = db.similarity_search(query, k=2) # Ищем 2 самых похожих куска
    
    print("\n=== НАЙДЕННЫЕ СОВПАДЕНИЯ В БАЗЕ ===")
    for i, doc in enumerate(docs):
        print(f"\n📄 Кусок №{i+1} (Блок: {doc.metadata.get('block')}):")
        print(doc.page_content[:300] + "...") # Выведем начало куска
        print("-" * 40)

if __name__ == "__main__":
    test_search()