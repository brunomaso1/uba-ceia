# RABBIT-SEND-RECIEVE-TEST

Aplicación de prueba para enviar y recibir mensajes a través de RabbitMQ usando Python con la librería `pika`. Esta aplicación se conecta a RabbitMQ, envía un mensaje a una cola y luego lo consume para verificar que el flujo de mensajes funciona correctamente.

# Test Instructions

1. Asegúrate de tener RabbitMQ instalado y en ejecución en tu máquina local.
2. Ejecuta el script `reciever.py` para iniciar el consumidor que escuchará los mensajes en la cola.
3. En otra terminal, ejecuta el script `send.py` para enviar un mensaje a la cola.
4. Observa la salida en la terminal del consumidor para verificar que el mensaje ha sido recibido correctamente.

Comando de ejecución:
```bash
python reciever.py
```
```bash
python send.py
```