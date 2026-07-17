from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from src import config

def get_llm():
    """ Инициализирует языковую модель GigaChat для генерации ответов. """
    return GigaChat(
        credentials=config.GIGACHAT_CREDENTIALS,
        scope="GIGACHAT_API_PERS",
        model="GigaChat",
        verify_ssl_certs=False,
        rquid=config.GIGACHAT_RQUID,
        temperature=0.2
    )

from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Возвращает бесплатную локальную модель эмбеддингов, работающую без интернета.
    """
    # cointegrated/rubert-tiny2 — отличная легкая модель, заточенная под русский язык
    model_name = "cointegrated/rubert-tiny2"
    
    print(f"Загрузка локальной модели эмбеддингов {model_name}...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'} # Будет работать на процессоре твоего ПК
    )

if __name__ == "__main__":
    print("Проверяем подключение к GigaChat...")
    try:
        llm = get_llm()
        res = llm.invoke("Привет! Ответь одним словом 'Работает', если ты меня слышишь.")
        print(f"Ответ от ИИ: {res.content}")
    except Exception as e:
        print(f"Что-то пошло не так: {e}")