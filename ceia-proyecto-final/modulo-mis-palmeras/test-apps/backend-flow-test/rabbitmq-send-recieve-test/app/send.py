# System imports

# Local imports

# Thrid party imports
import pika

HOST = "localhost"
PORT = 5672
VIRTUAL_HOST = "ai_vhost"
QUEUE_NAME = "hello"

connection = pika.BlockingConnection(pika.ConnectionParameters(host=HOST, port=PORT, virtual_host=VIRTUAL_HOST))
channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME)

body = "Hello World!"
channel.basic_publish(exchange="", routing_key=QUEUE_NAME, body=body)
print(f" [x] Sent '{body}'")

connection.close()
