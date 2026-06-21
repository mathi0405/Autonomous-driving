<#
.SYNOPSIS
    One-command pipeline: env setup, train PPO + SAC, evaluate vs PID/Stanley,
    build dashboard/figures/report, record a demo GIF. Requires Python 3.10-3.13.
.EXAMPLE
    .\run.ps1                      # fallback env, 100k steps + demo
    .\run.ps1 -Steps 300000        # train longer
    .\run.ps1 -Carla               # train on a running CARLA server (image obs)
    .\run.ps1 -SkipInstall         # reuse .venv, skip dependency install
    .\run.ps1 -SkipTrain           # just re-eval + rebuild dashboard/report/demo
#>
param(
    [int]$Steps = 100000,
    [string]$Cuda = "cu124",
    [switch]$Carla,
    [switch]$SkipInstall,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Py() { & python @args; if ($LASTEXITCODE -ne 0) { throw "python $($args -join ' ') failed" } }

function Find-Python {
    $old = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
    try {
        $supported = @("3.11", "3.12", "3.10", "3.13")
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($v in $supported) {
                try { & py "-$v" -c "import sys" 2>$null | Out-Null } catch {}
                if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
            }
        }
        foreach ($exe in @("python", "python3")) {
            if (Get-Command $exe -ErrorAction SilentlyContinue) {
                $v = $null
                try { $v = (& $exe -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))" 2>$null) } catch {}
                if ($supported -contains $v) { return ,@($exe) }
            }
        }
        foreach ($ver in @("311", "312", "310", "313")) {
            foreach ($root in @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles")) {
                $p = Join-Path $root "Python$ver\python.exe"
                if (Test-Path $p) { return ,@($p) }
            }
        }
        return $null
    } finally { $ErrorActionPreference = $old }
}

# --- 1. Virtual environment -------------------------------------------------
Step "Python virtual environment"
if (-not (Test-Path ".venv")) {
    $base = Find-Python
    if (-not $base) {
        Write-Host "No PyTorch-compatible Python (3.10-3.13) found." -ForegroundColor Red
        exit 1
    }
    Write-Host "Creating .venv with: $($base -join ' ')"
    $pyExe = $base[0]
    $pyArgs = if ($base.Count -gt 1) { @($base[1..($base.Count - 1)]) } else { @() }
    & $pyExe @pyArgs -m venv .venv
    if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "venv creation failed" -ForegroundColor Red; exit 1 }
}
. .\.venv\Scripts\Activate.ps1
& python --version

# --- 2. Dependencies --------------------------------------------------------
if (-not $SkipInstall) {
    Step "Installing dependencies (PyTorch + project)"
    & python -m pip install --upgrade pip
    try {
        & python -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
        if ($LASTEXITCODE -ne 0) { throw "cuda wheel" }
    } catch {
        Write-Warning "CUDA wheel '$Cuda' failed; installing default PyTorch."
        & python -m pip install torch
    }
    & python -m pip install -e ".[dev,viz]"
}

Step "GPU check"
& python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Restore the CUDA PyTorch build for CARLA, if dependency resolution replaced it with CPU.
if ($Carla) {
    $cudaOk = (& python -c "import torch;print(torch.cuda.is_available())") 2>$null
    if ("$cudaOk".Trim() -ne "True") {
        Write-Warning "Restoring CUDA PyTorch for CARLA GPU training..."
        & python -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda" --force-reinstall --no-deps
    }
}

# --- 3. Configure run target ------------------------------------------------
if ($Carla) { $envName = "carla";    $obs = "image"; $tag = "carla" }
else        { $envName = "fallback"; $obs = "state"; $tag = "fallback" }
$ppoRun = "ppo_$tag"
$sacRun = "sac_$tag"

# --- 4. Train ---------------------------------------------------------------
if (-not $SkipTrain) {
    Step "Training PPO ($envName/$obs, $Steps steps)"
    Py -m ad_rl.training.train --config configs/ppo.yaml --env $envName --obs $obs --total-timesteps $Steps --device auto --run-name $ppoRun
    Step "Training SAC ($envName/$obs, $Steps steps)"
    Py -m ad_rl.training.train --config configs/sac.yaml --env $envName --obs $obs --total-timesteps $Steps --device auto --run-name $sacRun
}

# --- 5. Evaluate ------------------------------------------------------------
Step "Evaluating agents"
Py -m ad_rl.evaluation.evaluate --model "runs/$ppoRun/best_model.zip" --algo ppo --env $envName --obs $obs --episodes 20 --agent-name PPO
Py -m ad_rl.evaluation.evaluate --model "runs/$sacRun/best_model.zip" --algo sac --env $envName --obs $obs --episodes 20 --agent-name SAC
Py -m ad_rl.evaluation.evaluate --model pid     --env $envName --obs $obs --episodes 20
Py -m ad_rl.evaluation.evaluate --model stanley --env $envName --obs $obs --episodes 20

# --- 6. Reports -------------------------------------------------------------
Step "Building dashboard, figures and report"
Py scripts/make_dashboard.py --results results/summary.json --out dashboard/index.html
Py scripts/make_figures.py
Py scripts/make_report.py

# --- 7. Demo ----------------------------------------------------------------
Step "Recording demo (PPO)"
Py scripts/record_demo.py --model "runs/$ppoRun/best_model.zip" --algo ppo --env $envName --obs $obs --episodes 3 --out docs/images/demo.gif

Step "Done"
Write-Host "Open:  dashboard\index.html   docs\images\demo.gif   docs\Autonomous_Driving_RL_Report.pdf" -ForegroundColor Green
Invoke-Item "dashboard\index.html"
