$ErrorActionPreference='Stop'
function admin{if(-not([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){Start-Process powershell -Verb runas "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"";exit}}
admin
$p='HKLM:\SOFTWARE\Microsoft\Windows\Dwm';$n='OverlayTestMode'
ni $p -Force>$null;sp $p $n 5 -Type DWord
$c=(gp $p -Name $n -EA SilentlyContinue).$n;echo "OverlayTestMode=$c"
$r=Read-Host 'Restart now? (Y/N)';if($r -match '^[Yy]'){Restart-Computer}
