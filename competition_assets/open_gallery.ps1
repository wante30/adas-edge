$ErrorActionPreference = 'Stop'
$galleryScript = Join-Path $PSScriptRoot 'tools\serve_gallery.py'
$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if (-not $pythonCommand) { throw 'Python 3 is required to open this gallery.' }
Start-Process -FilePath $pythonCommand.Source -ArgumentList @('"' + $galleryScript + '"') -WindowStyle Hidden
