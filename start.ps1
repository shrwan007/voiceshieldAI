#!/usr/bin/env pwsh
<#
.SYNOPSIS
    VoiceShield One-Click Setup & Launch Script
    Automatically configures PostgreSQL, runs migrations, and starts both servers.
#>

$ErrorActionPreference = "Continue"
$ROOT = "c:\Users\ASUS\OneDrive\Desktop\voiceshieldAI"

# ── Colour helpers ────────────────────────────────────────────────────────────
function Green($msg)  { Write-Host "✅ $msg" -ForegroundColor Green }
function Yellow($msg) { Write-Host "⏳ $msg" -ForegroundColor Yellow }
function Red($msg)    { Write-Host "❌ $msg" -ForegroundColor Red }
function Blue($msg)   { Write-Host "🔵 $msg" -ForegroundColor Cyan }

Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Blue "   VoiceShield — Setup & Launch"
Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Find PostgreSQL ───────────────────────────────────────────────────────────
$pgPaths = @(
    "C:\Program Files\PostgreSQL\18\bin",
    "C:\Program Files\PostgreSQL\17\bin",
    "C:\Program Files\PostgreSQL\16\bin",
    "C:\Program Files\PostgreSQL\15\bin",
    "C:\Program Files\PostgresPro\17\bin",
    "C:\Program Files\PostgresPro\16\bin"
)
$pgBin = $null
foreach ($p in $pgPaths) {
    if (Test-Path "$p\psql.exe") { $pgBin = $p; break }
}

if (-not $pgBin) {
    Red "PostgreSQL not found in standard locations."
    Write-Host "Please install PostgreSQL from https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
    Write-Host "Then re-run this script." -ForegroundColor Yellow
    exit 1
}
Green "Found PostgreSQL at: $pgBin"
$env:PATH = "$pgBin;$env:PATH"

# ── Wait for PostgreSQL service ───────────────────────────────────────────────
Yellow "Starting PostgreSQL service..."
$svc = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($svc) {
    if ($svc.Status -ne "Running") {
        Start-Service $svc.Name
        Start-Sleep 3
    }
    Green "PostgreSQL service running: $($svc.Name)"
} else {
    Yellow "No Windows service found — assuming PostgreSQL is running."
}

# ── Wait for pg_isready ───────────────────────────────────────────────────────
Yellow "Waiting for PostgreSQL to accept connections..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    $r = & "$pgBin\pg_isready.exe" -U postgres -q 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep 1
}
if (-not $ready) {
    Red "PostgreSQL is not ready. Check if it's running."
    exit 1
}
Green "PostgreSQL is ready."

# ── Create DB user & database ─────────────────────────────────────────────────
Yellow "Setting up database user and database..."
$env:PGPASSWORD = "voiceshield_pass"

# Try postgres superuser password
$superPass = "voiceshield_pass"   # matches the unattended install password
$env:PGPASSWORD = $superPass

# Create role (idempotent)
& "$pgBin\psql.exe" -U postgres -c @"
DO `$`$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'voiceshield') THEN
    CREATE ROLE voiceshield LOGIN PASSWORD 'voiceshield_pass';
  END IF;
END
`$`$;
"@ 2>$null

# Create database (idempotent)
$dbExists = & "$pgBin\psql.exe" -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='voiceshield_db'" 2>$null
if ($dbExists -ne "1") {
    & "$pgBin\psql.exe" -U postgres -c "CREATE DATABASE voiceshield_db OWNER voiceshield;" 2>$null
    Green "Created database: voiceshield_db"
} else {
    Green "Database voiceshield_db already exists."
}

# Grant privileges
& "$pgBin\psql.exe" -U postgres -d voiceshield_db -c "GRANT ALL PRIVILEGES ON DATABASE voiceshield_db TO voiceshield;" 2>$null
& "$pgBin\psql.exe" -U postgres -d voiceshield_db -c "GRANT ALL ON SCHEMA public TO voiceshield;" 2>$null
Green "Database privileges granted."

# ── Run Alembic migrations ────────────────────────────────────────────────────
Yellow "Running database migrations..."
Set-Location $ROOT
$result = & python -m alembic upgrade head 2>&1
if ($LASTEXITCODE -eq 0) {
    Green "Database migrations complete."
} else {
    Yellow "Migration output: $result"
    Yellow "If tables already exist, this is fine."
}

# ── Create storage directories ────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "$ROOT\audio_storage" | Out-Null
New-Item -ItemType Directory -Force -Path "$ROOT\models_cache" | Out-Null
Green "Storage directories ready."

# ── Find Node.js ──────────────────────────────────────────────────────────────
$nodePaths = @(
    "C:\Program Files\nodejs",
    "$env:APPDATA\npm",
    "$env:ProgramFiles\nodejs"
)
foreach ($p in $nodePaths) {
    if (Test-Path "$p\node.exe") {
        $env:PATH = "$p;$env:PATH"
        break
    }
}
$nodeVer = node --version 2>$null
if ($nodeVer) {
    Green "Node.js $nodeVer found."
} else {
    Red "Node.js not found. Please install from https://nodejs.org then re-run."
    exit 1
}

# ── Install npm packages ──────────────────────────────────────────────────────
if (-not (Test-Path "$ROOT\frontend\node_modules")) {
    Yellow "Installing frontend npm packages (first time, ~30s)..."
    Set-Location "$ROOT\frontend"
    npm install --silent 2>&1 | Out-Null
    Green "npm packages installed."
} else {
    Green "npm packages already installed."
}

# ── Launch backend ────────────────────────────────────────────────────────────
Blue ""
Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Blue "   Launching VoiceShield Servers"
Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

Set-Location $ROOT
Yellow "Starting backend on http://localhost:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$ROOT'
Write-Host '🛡️  VoiceShield Backend Starting...' -ForegroundColor Cyan
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"@ -WindowStyle Normal

Start-Sleep 3

# ── Launch frontend ───────────────────────────────────────────────────────────
Yellow "Starting frontend on http://localhost:5173 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$ROOT\frontend'
Write-Host '⚛️  VoiceShield Frontend Starting...' -ForegroundColor Cyan
node --version
npm run dev
"@ -WindowStyle Normal

Start-Sleep 4

# ── Open browser ─────────────────────────────────────────────────────────────
Blue ""
Green "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Green "   VoiceShield is starting up!"
Green "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""
Write-Host "  🌐 Dashboard  →  http://localhost:5173" -ForegroundColor White
Write-Host "  📖 API Docs   →  http://localhost:8000/docs" -ForegroundColor White
Write-Host "  ❤️  Health     →  http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "  Opening browser in 5 seconds..." -ForegroundColor Gray
Start-Sleep 5
Start-Process "http://localhost:5173"
Start-Process "http://localhost:8000/docs"
