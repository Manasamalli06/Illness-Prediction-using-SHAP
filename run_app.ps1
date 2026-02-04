$root = $PSScriptRoot

Write-Host "--- Checking Model ---"
if (-Not (Test-Path "$root/models/illness_risk_model.keras")) {
    Write-Host "Model not found. Training now..."
    Set-Location "$root/src/model"
    python train.py
} else {
    Write-Host "Model found. Skipping training. (Run src/model/train.py manually to retrain)"
}

Write-Host "--- Starting Web App ---"
Set-Location "$root/src/app"
python app.py
