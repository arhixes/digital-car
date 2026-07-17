import os
from langchain_qdrant import QdrantVectorStore
from langchain_gigachat import GigaChatEmbeddings
from src.ingestion.docx_parser import extract_text_from_docx

# Твой реальный авторизационный токен GigaChat (длинная строка из личного кабинета)
# ВСТАВЬ ЕГО СЮДА, если не используешь переменные окружения (.env)
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "MDE5ZjM2ZDUtYjEyNi03MjUxLTg1ZmUtMjllM2IwYzY2OTVmOjM1MWNjYjU3LTRiNWUtNGM4Zi1iY2VhLWM2MzI4MmUxYzk3ZQ==")

def main():
    docx_path = "data/raw/elantra_manual.docx"
    
    # 1. Извлекаем блоки
    chunks = extract_text_from_docx(docx_path)
    if not chunks:
        print("❌ Нет данных для индексации.")
        return

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # 2. Инициализируем новые официальные эмбеддинги
    print("🧠 Подключаем облачные эмбеддинги GigaChat...")
    embeddings = GigaChatEmbeddings(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS" # Используем персональный скоуп по умолчанию
    )

    # 3. Создаем базу данных Qdrant
    print("🗄️ Создаем базу данных Qdrant и отправляем векторы...")
    db = QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        path="data/qdrant_db",
        collection_name="elantra_manual",
        metadatas=metadatas
    )

    print("🎉 База данных успешно создана и заполнена векторами Сбера!")

if __name__ == "__main__":
    main()