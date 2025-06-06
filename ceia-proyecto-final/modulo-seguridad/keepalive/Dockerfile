FROM alpine:latest

RUN apk add --no-cache curl

COPY keepalive.sh /keepalive.sh
RUN chmod +x /keepalive.sh

CMD ["/keepalive.sh"]