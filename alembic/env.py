# ==============================================================================
# VoiceShield AI — Alembic Database Migration Environment
# ==============================================================================
# This module configures Alembic for both offline (SQL script generation) and
# online (direct database schema updates) migration execution using async SQLAlchemy
# with asyncpg driver support.
# ==============================================================================

import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ------------------------------------------------------------------------------
# 1. Path Setup and Model Discovery
# ------------------------------------------------------------------------------
# Ensure the root project directory is on sys.path for backend package imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Alembic Config object
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import the Base metadata from backend models to support autogenerate
try:
    from backend.models import Base
except ImportError:
    try:
        from backend.database import Base
    except ImportError:
        Base = None

target_metadata = Base.metadata if Base is not None else None


# ------------------------------------------------------------------------------
# 2. Database Connection URL Resolution
# ------------------------------------------------------------------------------
def get_url() -> str:
    """Retrieve database connection URL from environment variables, settings, or alembic.ini.

    Priority:
    1. DATABASE_URL environment variable
    2. backend.config.get_settings().database_url (if available)
    3. sqlalchemy.url from alembic.ini

    Returns:
        str: Resolved database connection string.
    """
    try:
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url

        try:
            from backend.config import get_settings
            settings = get_settings()
            if hasattr(settings, "database_url") and settings.database_url:
                return str(settings.database_url)
        except Exception:
            pass

        return config.get_main_option("sqlalchemy.url", "postgresql://voiceshield:voiceshield_pass@localhost:5432/voiceshield_db")
    except Exception as exc:
        sys.stderr.write(f"Warning: Failed to determine custom database URL: {exc}\n")
        return config.get_main_option("sqlalchemy.url", "postgresql://voiceshield:voiceshield_pass@localhost:5432/voiceshield_db")


# ------------------------------------------------------------------------------
# 3. Offline Migration Mode
# ------------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an
    Engine is acceptable here as well. By skipping the Engine creation we don't
    even need a DBAPI to be available. Calls to context.execute() here emit the
    given string to the script output.
    """
    url = get_url()
    # Offline mode typically runs via standard sync dialect without +asyncpg
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ------------------------------------------------------------------------------
# 4. Online Migration Mode (Async Engine via asyncpg)
# ------------------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    """Execute migration run against a synchronous connection wrapper.

    Args:
        connection (Connection): The active SQLAlchemy connection to run migrations on.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations within an async context.

    This function sets up the asyncpg engine, connects to the database,
    and delegates the migration execution to the synchronous callback.
    """
    configuration = config.get_section(config.config_ini_section) or {}
    url = get_url()

    # Ensure asyncpg driver is specified for the async engine
    if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    configuration["sqlalchemy.url"] = url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    except Exception as exc:
        sys.stderr.write(f"Error during async database migration: {exc}\n")
        raise
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Initializes an event loop to run async migrations via run_async_migrations.
    """
    try:
        asyncio.run(run_async_migrations())
    except Exception as exc:
        sys.stderr.write(f"Migration online execution failed: {exc}\n")
        raise


# ------------------------------------------------------------------------------
# 5. Mode Selection
# ------------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
