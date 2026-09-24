#!/usr/bin/env python3
"""
VoiceShield Database Setup Script
Waits for PostgreSQL to be ready, creates the DB user, database,
then runs Alembic migrations.
"""
import subprocess
import sys
import time
import os

PG_BIN = r"C:\Program Files\PostgreSQL\16\bin"
PSQL   = os.path.join(PG_BIN, "psql.exe")

def run(cmd, check=True, capture=False, env=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, capture_output=capture, text=True, env=e, shell=True)
    if check and r.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        if capture:
            print(r.stdout, r.stderr)
        sys.exit(1)
    return r

def wait_for_postgres(max_wait=60):
    """Poll pg_isready until PostgreSQL accepts connections."""
    pg_isready = os.path.join(PG_BIN, "pg_isready.exe")
    print("Waiting for PostgreSQL to start...", end="", flush=True)
    for _ in range(max_wait):
        r = subprocess.run([pg_isready, "-U", "postgres", "-q"],
                           capture_output=True)
        if r.returncode == 0:
            print(" ready!")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print("\n[ERROR] PostgreSQL did not start in time.")
    return False

def setup_db():
    env = {"PGPASSWORD": "voiceshield_pass"}

    # Create user (ignore if exists)
    print("Creating DB user 'voiceshield'...")
    run(f'"{PSQL}" -U postgres -c "DO $$ BEGIN '
        f"CREATE ROLE voiceshield LOGIN PASSWORD 'voiceshield_pass'; "
        f"EXCEPTION WHEN duplicate_object THEN NULL; END $$;\"",
        env={**os.environ, "PGPASSWORD": "voiceshield_pass"}, check=False)

    # Create database (ignore if exists)
    print("Creating database 'voiceshield_db'...")
    run(f'"{PSQL}" -U postgres -c '
        f'"SELECT 1 FROM pg_database WHERE datname=\'voiceshield_db\'" | '
        f'findstr /C:"1 row" > nul || '
        f'"{PSQL}" -U postgres -c "CREATE DATABASE voiceshield_db OWNER voiceshield;"',
        env={**os.environ, "PGPASSWORD": "voiceshield_pass"}, check=False)

    # Grant privileges
    run(f'"{PSQL}" -U postgres -d voiceshield_db -c '
        f'"GRANT ALL PRIVILEGES ON DATABASE voiceshield_db TO voiceshield;"',
        env={**os.environ, "PGPASSWORD": "voiceshield_pass"}, check=False)

    # Grant schema privileges  
    run(f'"{PSQL}" -U postgres -d voiceshield_db -c '
        f'"GRANT ALL ON SCHEMA public TO voiceshield;"',
        env={**os.environ, "PGPASSWORD": "voiceshield_pass"}, check=False)

    print("Database setup complete!")

def run_migrations():
    print("Running Alembic migrations...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=r"c:\Users\ASUS\OneDrive\Desktop\voiceshieldAI",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[WARNING] Alembic error:", result.stderr)
        print("Tables may need to be created manually via: alembic upgrade head")
    else:
        print("Migrations complete!")

if __name__ == "__main__":
    if not wait_for_postgres():
        sys.exit(1)
    setup_db()
    run_migrations()
    print("\n✅ Database ready! You can now start the backend.")
