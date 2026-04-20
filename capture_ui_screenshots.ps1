param(
    [string]$Root = (Get-Location).Path,
    [string]$FrontendRoot = "C:\Users\Shreya\.antigravity\frontend"
)

$ErrorActionPreference = "Stop"

$python = "C:\Users\Shreya\AppData\Local\Programs\Python\Python313\python.exe"
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$assetDir = Join-Path $Root "report_assets"
$profile = Join-Path $Root "chrome-headless-profile"
$browserOutput = Join-Path $assetDir "browser_ui_upload.png"
$streamlitOutput = Join-Path $assetDir "streamlit_ui_upload.png"

if (!(Test-Path $assetDir)) {
    New-Item -ItemType Directory -Path $assetDir | Out-Null
}
if (!(Test-Path $profile)) {
    New-Item -ItemType Directory -Path $profile | Out-Null
}

$htmlPath = (Resolve-Path (Join-Path $FrontendRoot "index.html")).Path
$htmlUri = [System.Uri]::new($htmlPath).AbsoluteUri

& $chrome '--headless=new' '--disable-gpu' '--no-first-run' "--user-data-dir=$profile" '--window-size=1440,1200' "--screenshot=$browserOutput" $htmlUri

$env:YOLO_CONFIG_DIR = Join-Path $Root "Ultralytics"
$streamlitArgs = @(
    "-m", "streamlit", "run", "app.py",
    "--server.headless", "true",
    "--server.address", "127.0.0.1",
    "--server.port", "8501",
    "--browser.gatherUsageStats", "false"
)

$job = Start-Job -ArgumentList $Root, $python, $streamlitArgs -ScriptBlock {
    param($JobRoot, $JobPython, $JobArgs)
    Set-Location $JobRoot
    $env:YOLO_CONFIG_DIR = Join-Path $JobRoot "Ultralytics"
    & $JobPython @JobArgs
}

try {
    $ready = $false
    for ($i = 0; $i -lt 30; $i += 1) {
        Start-Sleep -Seconds 2
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8501" -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
        }
    }

    if (-not $ready) {
        throw "Streamlit app did not become ready in time."
    }

    Start-Sleep -Seconds 3
    & $chrome '--headless=new' '--disable-gpu' '--no-first-run' "--user-data-dir=$profile" '--window-size=1440,1400' '--virtual-time-budget=15000' "--screenshot=$streamlitOutput" 'http://127.0.0.1:8501'
} finally {
    if ($job) {
        Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null
    }
}

Write-Host "Saved UI screenshots:"
Write-Host "- $browserOutput"
Write-Host "- $streamlitOutput"
