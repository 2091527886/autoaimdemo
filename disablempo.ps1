#requires -version 5.1
<#
  Toggle DWM MPO (OverlayTestMode = 5) / Revert / Check status

  Usage examples:
    - Disable MPO:       .\disablempo.ps1
    - Check only:        .\disablempo.ps1 -CheckOnly
    - Revert to default: .\disablempo.ps1 -Revert
    - No reboot prompt:  .\disablempo.ps1 -NoReboot
    - Force reboot:      .\disablempo.ps1 -ForceReboot

  Notes:
    - Writes HKLM:\SOFTWARE\Microsoft\Windows\Dwm\OverlayTestMode = 5 to disable MPO
    - Some systems/drivers require reboot or logoff to take effect
    - Uses 64-bit registry view and validates after write
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [switch]$Revert,
    [switch]$NoReboot,
    [switch]$CheckOnly,
    [switch]$ForceReboot
)

#region Helpers
function Test-ProcessIsElevated {
    try {
        $wid = [Security.Principal.WindowsIdentity]::GetCurrent()
        $prp = New-Object Security.Principal.WindowsPrincipal($wid)
        return $prp.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
    } catch { return $false }
}

function Relaunch-SelfAsAdmin {
    param([string[]]$PassthroughArgs)
    Write-Host "INFO: requesting administrator privileges..."
    $exe = (Get-Process -Id $PID).Path
    $psi = New-Object System.Diagnostics.ProcessStartInfo $exe
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $($PassthroughArgs -join ' ')"
    $psi.Verb = "runas"
    try { [Diagnostics.Process]::Start($psi) | Out-Null } catch { throw "User declined or elevation failed: $($_.Exception.Message)" }
    exit
}

function Get-ArgLineFromBoundParams {
    param([hashtable]$Bound)
    $pairs = @()
    foreach ($kv in $Bound.GetEnumerator()) {
        if ($kv.Value -is [switch]) {
            if ($kv.Value.IsPresent) { $pairs += "-$($kv.Key)" }
        } else {
            $escaped = $kv.Value.ToString().Replace('"','\"')
            $pairs += "-$($kv.Key) `"$escaped`""
        }
    }
    return ($pairs -join ' ')
}

function Get-OverlayTestModeValue {
    # use 64-bit registry view
    $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    $sub  = $base.OpenSubKey('SOFTWARE\\Microsoft\\Windows\\Dwm', $false)
    if ($null -eq $sub) { return $null }
    return $sub.GetValue('OverlayTestMode')
}

function Set-OverlayTestModeValue {
    param([int]$Value)
    $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    $sub  = $base.CreateSubKey('SOFTWARE\\Microsoft\\Windows\\Dwm')
    $sub.SetValue('OverlayTestMode', $Value, [Microsoft.Win32.RegistryValueKind]::DWord)
}

function Remove-OverlayTestModeValue {
    $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    $sub  = $base.OpenSubKey('SOFTWARE\\Microsoft\\Windows\\Dwm', $true)
    if ($sub -ne $null) { try { $sub.DeleteValue('OverlayTestMode', $true) } catch {} }
}

function Show-CurrentState {
    $val = Get-OverlayTestModeValue
    if ($null -ne $val) {
        Write-Host "STATUS: OverlayTestMode = $val"
        if ($val -eq 5) { Write-Host "STATUS: MPO is DISABLED (OverlayTestMode=5)" }
    } else {
        Write-Host "STATUS: OverlayTestMode is NOT set (system default)"
    }
}
#endregion

# Elevation
if (-not (Test-ProcessIsElevated)) {
    $argsLine = Get-ArgLineFromBoundParams -Bound $MyInvocation.BoundParameters
    Relaunch-SelfAsAdmin -PassthroughArgs $argsLine.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
}

try {
    if ($CheckOnly) {
        Show-CurrentState
        return
    }

    if ($Revert) {
        if ($PSCmdlet.ShouldProcess('HKLM:SOFTWARE\\Microsoft\\Windows\\Dwm', 'Remove OverlayTestMode')) {
            Remove-OverlayTestModeValue
            $readback = Get-OverlayTestModeValue
            if ($null -eq $readback) {
                Write-Host "OK: OverlayTestMode removed (restored default)"
            } else {
                Write-Warning "WARN: remove failed, still reading value: $readback"
            }
        }
    } else {
        if ($PSCmdlet.ShouldProcess('HKLM:SOFTWARE\\Microsoft\\Windows\\Dwm', 'Set OverlayTestMode=5 (disable MPO)')) {
            Set-OverlayTestModeValue -Value 5
            $readback = Get-OverlayTestModeValue
            if ($readback -eq 5) {
                Write-Host "OK: OverlayTestMode set to 5 (MPO disabled)"
            } else {
                Write-Warning "WARN: validation failed after write, read back: $readback"
            }
        }
    }

    Show-CurrentState
} catch {
    Write-Error "ERROR: $($_.Exception.Message)"
    exit 1
}

# Reboot logic
if ($ForceReboot) {
    Write-Host "INFO: rebooting now to apply changes..."
    Restart-Computer -Force
} elseif (-not $NoReboot) {
    $r = Read-Host "Reboot now to apply changes? (Y/N)"
    if ($r -match '^[Yy]') {
        Restart-Computer -Force
    } else {
        Write-Host "INFO: please logoff or reboot manually to fully apply the setting."
    }
}