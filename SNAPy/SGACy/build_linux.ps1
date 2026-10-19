# Build graph/geom Linux .so in Docker (glibc bookworm, x86_64).
# Does not touch PyPI. Leaves Windows .pyd and Darwin .so in place.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$versions = if ($args.Count) { $args } else { @("3.11", "3.12") }
foreach ($py in $versions) {
    $tag = "sgacy-linux-$($py.Replace('.', ''))"
    Write-Host "=== building image $tag (Python $py) ==="
    docker build --build-arg "PY=$py" -t $tag .
    Write-Host "=== compiling in $tag ==="
    docker run --rm -v "${PSScriptRoot}:/src" $tag
}
Write-Host "=== Linux artifacts ==="
Get-ChildItem -File | Where-Object { $_.Name -match '\.so$' } | Select-Object Name, Length
