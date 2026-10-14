# Pinned upstream checkout. Run from the CrowdWeave26 repository root.
param([string]$Destination = "ultralytics-src", [string]$Python = "python")
$ErrorActionPreference = "Stop"
$Manifest = Get-Content "upstream.json" -Raw | ConvertFrom-Json
if (Test-Path $Destination) { throw "Destination already exists: $Destination" }
& git clone $Manifest.repository $Destination
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
Push-Location $Destination
try {
    & git checkout --detach $Manifest.commit
    if ($LASTEXITCODE -ne 0) { throw "checkout failed" }
    & $Python -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    & $Python -c "from ultralytics import YOLO; YOLO('yolo26n.yaml').info()"
    if ($LASTEXITCODE -ne 0) { throw "model smoke test failed" }
    Write-Host "Pinned upstream: $($Manifest.commit)"
} finally { Pop-Location }
