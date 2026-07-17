import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_RQUID = os.getenv("GIGACHAT_RQUID")

# Путь к локальной папке с базой данных внутри нашего проекта
BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = os.path.join(BASE_DIR, "data", "qdrant_db")