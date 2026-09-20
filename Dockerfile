# Базовый легковесный образ с установленным Python
FROM python:3.10-slim

# Создаем рабочую директорию внутри сервера
WORKDIR /code

# Копируем файл зависимостей и устанавливаем их
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Копируем весь остальной код проекта в контейнер
COPY . .

# Открываем порт для Gradio-интерфейса
EXPOSE 7860

# Запускаем веб-приложение (app.py поднимает Gradio-сервер на 0.0.0.0:$PORT)
CMD ["python", "app.py"]
