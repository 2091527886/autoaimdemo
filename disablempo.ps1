#requires -version 5.1
<# 
  关闭 DWM MPO（OverlayTestMode = 5）/ 还原
  - 无参数：设置 HKLM:\SOFTWARE\Microsoft\Windows\Dwm\OverlayTestMode = 5
  - -Revert   ：删除 OverlayTestMode（还原系统默认）
  - -NoReboot ：不提示重启
#>

param(
    [switch]$Revert,
    [switch]$NoReboot
)

function Ensure-Elevated {
    $wid = [Security.Principal.WindowsIdentity]::GetCurrent()
    $prp = New-Object Security.Principal.WindowsPrincipal($wid)
    if (-not $prp.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
        Write-Host "[i] 正在请求管理员权限..."
        $psi = New-Object System.Diagnostics.ProcessStartInfo "powershell"
        $argsFromParams = ($MyInvocation.BoundParameters.GetEnumerator() | ForEach-Object { 
            if ($_.Value -is [switch]) { "-$($_.Key)" } else { "-$($_.Key) `"$($_.Value)`"" }
        }) -join " "
        $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $argsFromParams"
        $psi.Verb = "runas"
        try {
            [Diagnostics.Process]::Start($psi) | Out-Null
        } catch {
            Write-Error "用户拒绝提权或无法提权。"
        }
        exit
    }
}

function Set-OverlayTestMode5 {
    $path = 'HKLM:\SOFTWARE\Microsoft\Windows\Dwm'
    New-Item -Path $path -Force | Out-Null
    New-ItemProperty -Path $path -Name 'OverlayTestMode' -PropertyType DWord -Value 5 -Force | Out-Null
    Write-Host "[+] 已设置 $path\OverlayTestMode = 5（关闭 MPO）"
}

function Remove-OverlayTestMode {
    $path = 'HKLM:\SOFTWARE\Microsoft\Windows\Dwm'
    if (Test-Path $path) {
        if (Get-ItemProperty -Path $path -Name 'OverlayTestMode' -ErrorAction SilentlyContinue) {
            Remove-ItemProperty -Path $path -Name 'OverlayTestMode' -Force
            Write-Host "[+] 已删除 $path\OverlayTestMode（还原默认）"
        } else {
            Write-Host "[i] 未找到 $path\OverlayTestMode，无需还原。"
        }
    } else {
        Write-Host "[i] 未找到 $path，无需还原。"
    }
}

function Show-CurrentState {
    $path = 'HKLM:\SOFTWARE\Microsoft\Windows\Dwm'
    $val = (Get-ItemProperty -Path $path -Name 'OverlayTestMode' -ErrorAction SilentlyContinue).OverlayTestMode
    if ($null -ne $val) {
        Write-Host "[=] 当前 OverlayTestMode = $val"
        if ($val -eq 5) { Write-Host "[=] 状态：MPO 关闭" }
    } else {
        Write-Host "[=] 当前未设置 OverlayTestMode（系统默认）"
    }
}

Ensure-Elevated

try {
    if ($Revert) {
        Remove-OverlayTestMode
    } else {
        Set-OverlayTestMode5
    }
    Show-CurrentState
} catch {
    Write-Error "操作失败：$($_.Exception.Message)"
    exit 1
}

if (-not $NoReboot) {
    $r = Read-Host "是否现在重启计算机以生效? (Y/N)"
    if ($r -match '^[Yy]') {
        Restart-Computer -Force
    } else {
        Write-Host "[i] 请手动注销或重启使设置完全生效。"
    }
}
