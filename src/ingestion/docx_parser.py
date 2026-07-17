import os
from docx import Document

def parse_table_to_markdown(table) -> str:
    markdown_rows = []
    
    # Читаем все строки таблицы
    for i, row in enumerate(table.rows):
        # Собираем текст из каждой ячейки, убирая лишние пробелы и переносы
        cells_text = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        cleaned_cells = []
        for idx, text in enumerate(cells_text):
            if idx > 0 and text == cells_text[idx - 1] and row.cells[idx]._tc == row.cells[idx - 1]._tc:
                cleaned_cells.append("")
            else:
                cleaned_cells.append(text)
                
        markdown_row = "| " + " | ".join(cleaned_cells) + " |"
        markdown_rows.append(markdown_row)
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(cleaned_cells)) + " |"
            markdown_rows.append(separator)
            
    return "\n".join(markdown_rows)

def extract_text_from_docx(file_path: str, block_size: int = 1500) -> list:
    """Читает документ, сохраняя структуру текста и таблиц, и бьет на блоки"""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл не найден по пути {file_path}")
        return []

    print(f"Открываем документ: {file_path}")
    doc = Document(file_path)
    
    elements = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            from docx.text.paragraph import Paragraph
            p = Paragraph(element, doc)
            text = p.text.strip()
            if text:
                elements.append(text)
                
        elif element.tag.endswith('tbl'):
            from docx.table import Table
            t = Table(element, doc)
            table_md = parse_table_to_markdown(t)
            if table_md:
                elements.append(f"\n[ТАБЛИЦА ДАННЫХ]:\n{table_md}\n")

   # Теперь собираем элементы в блоки поменьше, чтобы темы не перемешивались
    print("Нарезаем текст и таблицы на смысловые блоки с перекрытием..")
    chunks = []
    current_chunk = []
    current_length = 0
    block_counter = 1
    
    OPTIMAL_BLOCK_SIZE = 1000 
    # Количество элементов, которые будут дублироваться в следующем блоке
    OVERLAP_ELEMENTS_COUNT = 3 

    for i, item in enumerate(elements):
        item_len = len(item)

        # Переносим блок, если превысили лимит
        if current_length + item_len > OPTIMAL_BLOCK_SIZE and current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "metadata": {"block": block_counter, "source": os.path.basename(file_path)}
            })
            block_counter += 1
            
            # берем последние N элементов текущего блока для начала следующего
            current_chunk = current_chunk[-OVERLAP_ELEMENTS_COUNT:] if len(current_chunk) >= OVERLAP_ELEMENTS_COUNT else current_chunk
            current_length = sum(len(x) for x in current_chunk)
            
        current_chunk.append(item)
        current_length += item_len

    if current_chunk:
        chunks.append({
            "text": "\n".join(current_chunk),
            "metadata": {"block": block_counter, "source": os.path.basename(file_path)}
        })

    print(f"Успешно извлечено блоков с перекрытием: {len(chunks)}")
    return chunks