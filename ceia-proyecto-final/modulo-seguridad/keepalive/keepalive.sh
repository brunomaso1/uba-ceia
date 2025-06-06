#!/bin/sh

TARGET_URL=${TARGET_URL:-http://192.168.0.3}
PING_INTERVAL=${PING_INTERVAL:-3600}

TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}

send_telegram_alert() {
  MSG="$1"
  if [ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    echo "[WARN] Sending Telegram alert: $MSG"
    curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/sendMessage" \
      -d chat_id="$TELEGRAM_CHAT_ID" \
      -d text="$MSG"
  fi
}

echo "Keepalive script started. Pinging $TARGET_URL every $PING_INTERVAL seconds."

while true; do
  echo "[INFO] Sending request to $TARGET_URL - $(date)"
  if ! curl -s --max-time 10 "$TARGET_URL" > /dev/null; then
    send_telegram_alert "⚠️ KeepAlive: No se pudo contactar con $TARGET_URL a las $(date)"
  fi
  sleep "$PING_INTERVAL"
done
