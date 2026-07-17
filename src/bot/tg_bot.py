import os
import telebot
import qdrant_client
from langchain_gigachat import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from src.ingestion.docx_parser import extract_text_from_docx
import requests
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from telebot.types import ReplyKeyboardMarkup, KeyboardButton

#  КОНФИГУРАЦИЯ И АВТОРИЗАЦИЯ
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ТОКЕН")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "ТОКЕН")
QDRANT_PATH = "data/qdrant_db"
chat_histories = {}

bot = telebot.TeleBot(BOT_TOKEN)
print("Загружаем мануал и инициализируем поисковые движки...")
all_chunks = extract_text_from_docx("data/raw/elantra_manual.docx")

def get_search_engines(chunks):
    embeddings = GigaChatEmbeddings(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS"
    )
    client = qdrant_client.QdrantClient(path=QDRANT_PATH)
    db = QdrantVectorStore(
        client=client,
        collection_name="elantra_manual",
        embedding=embeddings
    )
    qdrant_engine = db.as_retriever(search_kwargs={"k": 4})
    
    
    @bot.message_handler(commands=['clear', 'reset'])
    @bot.message_handler(func=lambda message: message.text == "🧹 Сбросить контекст")
    def reset_context_handler(message):
        """Очищает историю диалога с Михалычем"""
        chat_id = message.chat.id
        
        # Удаляем историю, если она есть
        if chat_id in chat_histories:
            del chat_histories[chat_id]
            
        markup = ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(KeyboardButton("🧹 Сбросить контекст"))
        
        bot.send_message(
            chat_id, 
            "Так, ладно, давай начнем с чистого листа. Про что там у тебя? Рассказывай, ласточку починим!", 
            reply_markup=markup
        )

    # БМ25
    langchain_docs = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"]) 
        for chunk in chunks
    ]
    bm25_engine = BM25Retriever.from_documents(langchain_docs)
    bm25_engine.k = 3
    
    return qdrant_engine, bm25_engine

qdrant_retriever, bm25_retriever = get_search_engines(all_chunks)


# рерайт
def rewrite_query(user_question):
    llm = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        temperature=0.2
    )
    
    system_instruction = (
        "Ты — технический ассистент автомеханика. Твоя задача — взять короткий вопрос пользователя "
        "и дополнить его ТОЛЬКО прямыми техническими синонимами (например, трубки -> магистрали, шланги). "
        "СТРОГО ЗАПРЕЩЕНО: додумывать за пользователя контекст, добавлять конкретные системы автомобиля "
        "(охлаждение, кондиционер, ГРМ), если они явно не указаны в вопросе. "
        "Выдай ТОЛЬКО список ключевых слов, разделенных ОБЫЧНЫМИ ПРОБЕЛАМИ. Ни в коем случае не склеивай слова вместе."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Оригинальный вопрос: {user_question}")
        ])
        return response.content.strip()
    except Exception as e:
        print(f"Ошибка рерайта запроса: {e}, используем оригинал.")
        return user_question

def generate_final_answer(chat_id: int, context: str, user_question: str) -> str:
    """Генерирует финальный ответ от лица Михалыча с учетом истории диалога и требований GigaChat API"""
    llm = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS"
    )
    
    if chat_id not in chat_histories:
        system_prompt = (
            "Ты — Михалыч AI, опытный, суровый, но толковый и дружелюбный автомеханик из соседнего гаража. "
            "Ты идеально знаешь технический мануал для автомобиля Hyundai Elantra 2004 года.\n\n"
            "ПРАВИЛА ПОВЕДЕНИЯ:\n"
            "1. Отвечай СТРОГО на основе предоставленного контекста из мануала.\n"
            "2. Если контекст пустой или тебе не хватает конкретики для точного ответа (например, пользователь "
            "спросил про абстрактный 'момент затяжки гайки' или 'замену ремня', не уточнив какой именно узел), "
            "НЕ придумывай ничего от себя и не галлюцинируй. Вместо этого задай вежливый, но чисто механический "
            "уточняющий вопрос от лица Михалыча (например: 'Слушай, земляк, а гайка-то какая? Колесная, ступицы или ГБЦ? "
            "Уточни, и я гляну в книгу').\n"
            "3. Общайся в характерном разговорном стиле гаражного мастера (используй словечки вроде 'земляк', 'ласточка', 'гляну в книгу', 'кардан мне в дышло'), но без мата и грубости."
        )
        chat_histories[chat_id] = [SystemMessage(content=system_prompt)]
    
    combined_user_message = (
        f"ИСПОЛЬЗУЙ ЭТОТ КОНТЕКСТ ИЗ МАНУАЛА ДЛЯ ОТВЕТА:\n"
        f"=== НАЧАЛО КОНТЕКСТА ===\n"
        f"{context}\n"
        f"=== КОНЕЦ КОНТЕКСТА ===\n\n"
        f"Оригинальный вопрос пользователя: {user_question}"
    )
    
    chat_histories[chat_id].append(HumanMessage(content=combined_user_message))
    
    if len(chat_histories[chat_id]) > 8:
        chat_histories[chat_id] = [chat_histories[chat_id][0]] + chat_histories[chat_id][-6:]
        
    try:

        response = llm.invoke(chat_histories[chat_id])
        ai_response_text = response.content
        chat_histories[chat_id][-1] = HumanMessage(content=user_question)
        chat_histories[chat_id].append(AIMessage(content=ai_response_text))
        
        return ai_response_text
        
    except Exception as e:
        print(f"⚠️ Ошибка генерации ответа GigaChat: {e}")
        # Если произошла ошибка, лучше удалить последнее добавленное сообщение, чтобы не ломать историю
        if len(chat_histories[chat_id]) > 1:
            chat_histories[chat_id].pop()
        return "Что-то у меня в гараже свет моргнул... Повтори вопрос, земляк, не расслышал."

def tool_maintenance_scheduler(current_odometer: int) -> str:
    try:
        # Интервал обслуживания
        interval = 10000
        next_to = ((current_odometer // interval) + 1) * interval
        km_left = next_to - current_odometer
    
        if km_left < 500:
            status = "⚠️ ВНИМАНИЕ: Срочно планируй визит в гараж! Пора менять масло и расходники."
        else:
            status = "Статус: Всё в рамках допуска. Спокойно ездим дальше."
            
        result = (
            f"📊 **Анализ периодичности ТО:**\n"
            f"🔹 Текущий пробег: {current_odometer:,} км\n"
            f"🔹 Следующее плановое ТО: **{next_to:,} км**\n"
            f"🏁 Осталось проехать: **{km_left:,} км**\n\n"
            f"{status}\n"
            f"📅 *Запись создана автоматически. Напомню тебе ближе к делу!*"
        )
        return result
    except Exception as e:
        return f"Не удалось рассчитать ТО: {e}"

def extract_city_with_llm(text: str) -> str:
    """Использует GigaChat, чтобы вытащить название города из запроса на русском или английском."""
    llm = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        temperature=0.0  
    )
    
    system_prompt = (
        "Твоя задача — проанализировать текст пользователя и найти в нем название населенного пункта (города, села и т.д.). "
        "Выведи ТОЛЬКО ОДНО слово — название этого города в начальной форме (например, 'Москва', 'Саратов', 'Казань'). "
        "Если в тексте нет упоминания какого-либо города, выведи ровно одно слово: Saratov"
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ])
        city = response.content.strip().replace(".", "").replace(",", "")
        return city if city else "Saratov"
    except Exception as e:
        print(f"⚠️ Ошибка определения города через LLM: {e}")
        return "Saratov"
    
def tool_weather_control(city: str = "Saratov") -> str:
    """
    Запрашивает погоду через Open-Meteo API (без ключей), анализирует температуру
    и осадки для рекомендаций по шинам и мойке автомобиля.
    """
    try:
        #  координаты города 
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=ru&format=json"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return f"⚠️ Не смог найти город '{city}' для проверки погоды."
            
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        city_name = geo_res["results"][0]["name"]

        # прогноз погоды на сегодня и завтра
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
        weather_res = requests.get(weather_url).json()
        
        # Среднесуточная температура
        temp_max = weather_res["daily"]["temperature_2m_max"][0]
        temp_min = weather_res["daily"]["temperature_2m_min"][0]
        avg_temp = (temp_max + temp_min) / 2
        
        # Осадки на сегодня и завтра 
        rain_today = weather_res["daily"]["precipitation_sum"][0]
        rain_tomorrow = weather_res["daily"]["precipitation_sum"][1]
        
        # Логика по шинам
        if avg_temp < 7.0:
            tyre_recommendation = (
                "❄️ **Шины:** Среднесуточная температура ниже +7°C (около " f"{avg_temp:.1f}°C). "
                "Если еще не переобулся на **зимнюю резину** — пора! Летняя резина при такой температуре дубеет и теряет сцепление."
            )
        else:
            tyre_recommendation = (
                "☀️ **Шины:** Среднесуточная температура стабильно выше +7°C (около " f"{avg_temp:.1f}°C). "
                "Можно спокойно гонять на **летней резине**. Если всё еще на шипах — переобувайся, не точи асфальт."
            )
            
        # Логика по мойке
        if rain_today > 1.0:
            wash_recommendation = "🚗💦 **Мойка:** На сегодня передают дождь/осадки. На мойку заезжать **смысла нет**, только деньги выкинешь."
        elif rain_tomorrow > 1.0:
            wash_recommendation = "🚗🌤️ **Мойка:** Сегодня сухо, но на завтра обещают дожди. Рекомендую **подождать с мойкой**, кузов быстро испачкается."
        else:
            wash_recommendation = "🚗✨ **Мойка:** Погода шепчет! Осадков ни сегодня, ни завтра не предвидится. **Отличный момент, чтобы помыть ласточку!**"

        result = (
            f"🌤️ **Контроль погоды для г. {city_name}:**\n"
            f"🌡️ Температура: от {temp_min:.1f}°C до {temp_max:.1f}°C (средняя ~{avg_temp:.1f}°C)\n"
            f"☔ Осадки сегодня: {rain_today} мм | Завтра: {rain_tomorrow} мм\n\n"
            f"{tyre_recommendation}\n\n"
            f"{wash_recommendation}"
        )
        return result
        
    except Exception as e:
        return f"⚠️ Не удалось получить данные о погоде: {e}"
    
# ОБРАБОТЧИК ДИАЛОГА В TELEGRAM
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_question = message.text.strip()
    print(f"\n📩 [ВХОДЯЩИЙ ВОПРОС]: '{user_question}'")

    if any(keyword in user_question.lower() for keyword in ["то", "пробег", "обслуживание", "напомни"]):
        numbers = [int(s) for s in user_question.split() if s.isdigit()]
        if numbers:
            odometer = numbers[0]
            if 1000 < odometer < 999999:
                print("🛠️ [ИНСТРУМЕНТ]: Запуск Maintenance_Scheduler...")
                ai_response = tool_maintenance_scheduler(odometer)
                bot.reply_to(message, ai_response, parse_mode='Markdown')
                print("✅ Ответ инструмента отправлен!")
                return

    if any(keyword in user_question.lower() for keyword in ["погода", "мойка", "мыть", "дождь", "шина", "шины", "переобуть"]):
        print("🛠️ [ИНСТРУМЕНТ]: Запуск Контроля погодных условий...")
        
        city = extract_city_with_llm(user_question)
        print(f"📍 [ОПРЕДЕЛЕН ГОРОД]: {city}")
                
        ai_response = tool_weather_control(city)
        bot.reply_to(message, ai_response, parse_mode='Markdown')
        print("✅ Ответ инструмента отправлен!")
        return

    # рерайт запроса
    optimized_query = rewrite_query(user_question)
    print(f"🧠 [РЕРАЙТ ЗАПРОСА]: '{optimized_query}'")
    
    # гибридный поиск
    docs_qdrant = qdrant_retriever.invoke(optimized_query)
    docs_bm25 = bm25_retriever.invoke(optimized_query)

    all_found_docs = docs_qdrant + docs_bm25
    
    # дедупликация
    unique_docs = []
    seen_texts = set()
    for doc in all_found_docs:
        if doc.page_content not in seen_texts:
            seen_texts.add(doc.page_content)
            unique_docs.append(doc)
            
    print(f"[ГИБРИДНЫЙ ПОИСК НАШЕЛ]: {len(unique_docs)} уникальных блоков.")
    
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    
    # генерация финального ответа
    print("Отправляем сборку в GigaChat к Михалычу...")
    ai_response = generate_final_answer(message.chat.id, context, user_question)
    
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton("🧹 Сбросить整合 контекст" if "🧹" not in message.text else "🧹 Сбросить контекст")) 
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton("🧹 Сбросить контекст"))

    # отправка в Telegram 
    bot.reply_to(message, ai_response, parse_mode='Markdown', reply_markup=markup)
    print("Ответ успешно отправлен!")

if __name__ == "__main__":
    print("🚀  Михалыч AI успешно запущен и готов к работе в гараже!")
    bot.infinity_polling()