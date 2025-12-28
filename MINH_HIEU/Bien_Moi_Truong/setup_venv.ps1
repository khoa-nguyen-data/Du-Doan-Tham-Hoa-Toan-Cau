# setup_venv.ps1
param(
  [string]$venvName = ".venv",
  [string]$kernelName = "du-doan-venv",
  [string]$displayName = "Python (du-doan-venv)"
)

$projectRoot = (Get-Location).Path
$venvPath = Join-Path $projectRoot $venvName
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPath)) {
  Write-Host "Creating venv at $venvPath"
  py -3 -m venv $venvPath
} else {
  Write-Host "Virtual environment already exists at $venvPath"
}

Write-Host "Upgrading pip and installing packages via $pythonExe"
& $pythonExe -m pip install --upgrade pip

$packages = @("pandas","numpy","matplotlib","seaborn","plotly","folium","notebook","ipykernel")
& $pythonExe -m pip install $packages

Write-Host "Registering Jupyter kernel as '$displayName' (name: $kernelName)"
& $pythonExe -m ipykernel install --user --name=$kernelName --display-name="$displayName"

Write-Host "Done. To use the venv now (CMD):"
Write-Host "  .\$venvName\Scripts\\activate.bat"
Write-Host "Or run the venv python directly:"
Write-Host "  .\$venvName\Scripts\python -c `"import pandas as pd; print(pd.__version__)`""