#requires -version 5.1
<#
  关闭 DWM MPO（OverlayTestMode = 5）/ 还原 / 检查状态

  用法示例：
    - 关闭 MPO：      .\disablempo.ps1
    - 仅检查：        .\disablempo.ps1 -CheckOnly
    - 还原：          .\disablempo.ps1 -Revert
    - 不提示重启：    .\disablempo.ps1 -NoReboot
    - 修改后强制重启：.\disablempo.ps1 -ForceReboot

  说明：
    - 写入 HKLM:\SOFTWARE\Microsoft\Windows\Dwm\OverlayTestMode = 5 以关闭 MPO
    - 某些系统/驱动版本需重启或注销后才完全生效
    - 本脚本将使用 64 位注册表视图进行读写并在操作后立即校验
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
    Write-Host "[i] 正在请求管理员权限..."
    $exe = (Get-Process -Id $PID).Path
    $psi = New-Object System.Diagnostics.ProcessStartInfo $exe
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $($PassthroughArgs -join ' ')"
    $psi.Verb = "runas"
    try { [Diagnostics.Process]::Start($psi) | Out-Null } catch { throw "用户拒绝提权或无法提权：$($_.Exception.Message)" }
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
    # 确保使用 64 位注册表视图
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
        Write-Host "[=] 当前 OverlayTestMode = $val"
        if ($val -eq 5) { Write-Host "[=] 状态：MPO 关闭（OverlayTestMode=5）" }
    } else {
        Write-Host "[=] 当前未设置 OverlayTestMode（系统默认）"
    }
}
#endregion

# 提权
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
        if ($PSCmdlet.ShouldProcess('HKLM:SOFTWARE\\Microsoft\\Windows\\Dwm', '移除 OverlayTestMode')) {
            Remove-OverlayTestModeValue
            $readback = Get-OverlayTestModeValue
            if ($null -eq $readback) {
                Write-Host "[+] 已删除 OverlayTestMode（还原默认）"
            } else {
                Write-Warning "[-] 删除失败：仍读取到值 $readback"
            }
        }
    } else {
        if ($PSCmdlet.ShouldProcess('HKLM:SOFTWARE\\Microsoft\\Windows\\Dwm', '设置 OverlayTestMode=5（关闭 MPO）')) {
            Set-OverlayTestModeValue -Value 5
            $readback = Get-OverlayTestModeValue
            if ($readback -eq 5) {
                Write-Host "[+] 已设置 OverlayTestMode = 5（关闭 MPO）"
            } else {
                Write-Warning "[-] 写入后校验失败：读取到 $readback"
            }
        }
    }

    Show-CurrentState
} catch {
    Write-Error "操作失败：$($_.Exception.Message)"
    exit 1
}

# 重启逻辑
if ($ForceReboot) {
    Write-Host "[i] 即将重启以应用更改..."
    Restart-Computer -Force
} elseif (-not $NoReboot) {
    $r = Read-Host "是否现在重启计算机以生效? (Y/N)"
    if ($r -match '^[Yy]') {
        Restart-Computer -Force
    } else {
        Write-Host "[i] 请手动注销或重启使设置完全生效。"
    }
}