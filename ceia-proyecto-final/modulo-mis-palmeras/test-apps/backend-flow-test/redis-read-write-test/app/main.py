# System imports
import os

# Local imports

# Third-party imports
from redis import Redis
from redis.exceptions import AuthenticationError, ConnectionError, RedisError, TimeoutError

REDIS_USER = os.getenv("REDIS_USER", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redispassword")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = f"redis://{REDIS_USER}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"


def write_test_data(key: str, value: str) -> bool:
    try:
        client = Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        return bool(client.set(name=key, value=value))
    except (AuthenticationError, ConnectionError, TimeoutError) as error:
        print(f"Redis write error ({type(error).__name__}): {error}")
        return False
    except RedisError as error:
        print(f"Redis write error (RedisError): {error}")
        return False


def read_test_data(key: str) -> str | None:
    try:
        client = Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        return client.get(name=key)
    except (AuthenticationError, ConnectionError, TimeoutError) as error:
        print(f"Redis read error ({type(error).__name__}): {error}")
        return None
    except RedisError as error:
        print(f"Redis read error (RedisError): {error}")
        return None


if __name__ == "__main__":
    test_key = "redis:test:key"
    test_value = "ok-from-python"

    write_ok = write_test_data(test_key, test_value)
    read_value = read_test_data(test_key)

    print(f"Write OK: {write_ok}")
    print(f"Read value: {read_value}")
