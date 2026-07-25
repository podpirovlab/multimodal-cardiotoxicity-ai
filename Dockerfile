# Базовый легковесный образ с установленным Python
FROM python:3.10-slim

# Создаем рабочую директорию внутри сервера
WORKDIR /code

# Копируем файл зависимостей и устанавливаем их
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Копируем весь остальной код проекта в контейнер
COPY . .

# Открываем порт для Gradio и запускаем наше монолитное приложение
CMD ["python", "advanced_model.py"]
