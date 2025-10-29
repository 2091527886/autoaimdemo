$ErrorActionPreference='Stop'
function admin{if(-not([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){Start-Process powershell -Verb runas "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"";exit}}
admin
$p='HKLM:\SOFTWARE\Microsoft\Windows\Dwm';$n='OverlayTestMode'
ni $p -Force>$null
try{Remove-ItemProperty $p $n -EA Stop;echo 'Removed'}catch{sp $p $n 0 -Type DWord;echo 'Set 0'}
$c=(gp $p -Name $n -EA SilentlyContinue).$n;if($null -eq $c){$c='<not set>'};echo "OverlayTestMode=$c"
$r=Read-Host 'Restart now? (Y/N)';if($r -match '^[Yy]'){Restart-Computer}
