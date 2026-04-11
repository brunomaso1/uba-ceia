# System imports
import os
from typing import Optional, AsyncGenerator

# Local imports

# Third-party imports
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgrespassword@postgres/postgresdb")

# Global pool instance
_pool: Optional[AsyncConnectionPool] = None


def get_database_url() -> str:
    """Get database URL from environment variables."""
    return DATABASE_URL


async def connect_db(database_url: str, min_size: int = 1, max_size: int = 20):
    """Initialize and open the connection pool."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(conninfo=database_url, min_size=min_size, max_size=max_size)
        await _pool.open()


async def close_db():
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_db_connection() -> AsyncGenerator[AsyncConnection, None]:
    """
    FastAPI dependency that provides a database connection.

    Usage:
        @app.get("/endpoint/")
        async def endpoint(conn = Depends(get_db_connection)):
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM table")
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call connect_db() first.")
    async with _pool.connection() as conn:
        yield conn
