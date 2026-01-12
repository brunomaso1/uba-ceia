# Must be executed with: . .\set-env.ps1
# (the first dot is important so variables remain in the current environment)

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile    = Join-Path $ScriptDir ".env.prod"

# Verify that it's being executed with dot-sourcing
if ($MyInvocation.InvocationName -notmatch '^\.') {
    Write-Host "El archivo debe ejecutarse utilizando '. .\set-env.ps1'"
    exit 1
}

# Verify file existence
if (-not (Test-Path $EnvFile)) {
    Write-Host "No existe el archivo $EnvFile."
    exit 1
}

Write-Host "Exportando variables de entorno del archivo $EnvFile."

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    # Ignore empty lines or comments
    if ($line -and -not ($line -match '^\s*#')) {
        $parts = $line -split '=', 2
        if ($parts.Length -eq 2) {
            $name  = $parts[0].Trim()
            $value = $parts[1].Trim()
            Write-Host $name
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Write-Host "Variables de entorno exportadas correctamente."
