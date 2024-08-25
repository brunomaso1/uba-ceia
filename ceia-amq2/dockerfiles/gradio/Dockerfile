FROM python:3.8-slim

WORKDIR /gradioapp

COPY requirements.txt /gradioapp/
RUN pip install -r requirements.txt

COPY . /gradioapp/

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "-u", "gradio_app.py"]