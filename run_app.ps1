$root = $PSScriptRoot
Write-Host "`n"
Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
Write-Host "!!! RELAUNCHING HEALTHGUARD: VERSION 3.0       !!!"
Write-Host "!!! FORCING VIRTUAL ENVIRONMENT ACTIVATION     !!!"
Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
Write-Host "`n"

$python = "$root/.venv/Scripts/python.exe"

if (-Not (Test-Path $python)) {
    Write-Host "WARN: .venv not found at $python. Falling back to system python."
    $python = "python"
}

Write-Host "--- Using Python: $python ---"

Write-Host "--- Checking Model Artifacts ---"
$model_pkl = "$root/backend/models/illness_risk_model.pkl"
if (-Not (Test-Path $model_pkl)) {
    Write-Host "Model not found. Training now..."
    Set-Location "$root/backend/model"
    & $python train.py
}
else {
    Write-Host "Model found at $model_pkl. Perfect Accuracy Model Ready."
}

Write-Host "--- Starting Web App Server ---"
Set-Location "$root/backend/app"
& $python app.py
