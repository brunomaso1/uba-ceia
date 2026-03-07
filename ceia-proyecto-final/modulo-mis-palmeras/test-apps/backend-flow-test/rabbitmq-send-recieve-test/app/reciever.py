# System imports
import sys, os

# Local imports

# Thrid party imports
import pika

HOST = "localhost"
PORT = 5672
VIRTUAL_HOST = "ai_vhost"
QUEUE_NAME = "hello"


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=HOST, port=PORT, virtual_host=VIRTUAL_HOST))
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME)

    def callback(ch, method, properties, body):
        print(f" [x] Received {body}")

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)

    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
