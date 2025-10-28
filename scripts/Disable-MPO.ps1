$ErrorActionPreference = 'Stop'

function Ensure-Admin {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "[INFO] Not running as Administrator. Relaunching elevated..." -ForegroundColor Yellow
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "powershell.exe"
        $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
        $psi.Verb = "runas"
        [Diagnostics.Process]::Start($psi) | Out-Null
        exit
    }
}

Ensure-Admin

$regPath = "HKLM:\SOFTWARE\Microsoft\Windows\Dwm"
$name = "OverlayTestMode"
$value = 5

New-Item -Path $regPath -Force | Out-Null
Set-ItemProperty -Path $regPath -Name $name -Type DWord -Value $value

Write-Host "[OK] Set $regPath\$name = $value (Disable MPO)" -ForegroundColor Green

$current = (Get-ItemProperty -Path $regPath -Name $name -ErrorAction SilentlyContinue).$name
Write-Host "[STATE] Current OverlayTestMode = $current"

Write-Host "`nA restart is required for the change to take effect." -ForegroundColor Yellow
$resp = Read-Host "Restart now? (Y/N)"
if ($resp -match '^[Yy]') {
    Restart-Computer
}
