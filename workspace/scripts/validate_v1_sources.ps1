$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
if ((Split-Path -Leaf $PSScriptRoot) -ne 'scripts') {
    $workspace = 'D:\workspace'
}
$raceRoot = Join-Path $workspace 'raceS9'
$specs = @(
    @{ Name = 'Concat'; Host = 'concat.cpp'; Tiling = 'concat_tiling.h'; Kernel = 'concat.cpp'; Symbol = 'concat' },
    @{ Name = 'Greater'; Host = 'greater.cpp'; Tiling = 'greater_tiling.h'; Kernel = 'greater.cpp'; Symbol = 'greater' },
    @{ Name = 'IndexAdd'; Host = 'index_add.cpp'; Tiling = 'index_add_tiling.h'; Kernel = 'index_add.cpp'; Symbol = 'index_add' },
    @{ Name = 'Transpose'; Host = 'transpose.cpp'; Tiling = 'transpose_tiling.h'; Kernel = 'transpose.cpp'; Symbol = 'transpose' },
    @{ Name = 'SquareSumV1'; Host = 'square_sum_v1.cpp'; Tiling = 'square_sum_v1_tiling.h'; Kernel = 'square_sum_v1.cpp'; Symbol = 'square_sum_v1' }
)

$errors = [System.Collections.Generic.List[string]]::new()
foreach ($spec in $specs) {
    $base = Join-Path $raceRoot $spec.Name
    $hostFile = Join-Path $base ('op_host\' + $spec.Host)
    $tiling = Join-Path $base ('op_host\' + $spec.Tiling)
    $kernel = Join-Path $base ('op_kernel\' + $spec.Kernel)
    foreach ($path in @($hostFile, $tiling, $kernel)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $errors.Add("missing: $path") }
    }
    if (Test-Path -LiteralPath $hostFile) {
        $hostText = Get-Content -LiteralPath $hostFile -Raw
        if ($hostText -notmatch "OP_ADD\($($spec.Name)\)") { $errors.Add("$($spec.Name): missing OP_ADD") }
        if ($hostText -notmatch 'AddConfig\("ascend910b"\)') { $errors.Add("$($spec.Name): missing ascend910b config") }
    }
    if (Test-Path -LiteralPath $kernel) {
        $kernelText = Get-Content -LiteralPath $kernel -Raw
        if ($kernelText -notmatch "void\s+$($spec.Symbol)\s*\(") { $errors.Add("$($spec.Name): missing kernel symbol $($spec.Symbol)") }
        if ($kernelText -notmatch 'GET_TILING_DATA') { $errors.Add("$($spec.Name): missing GET_TILING_DATA") }
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Host 'S9 V1 source structure and registration checks: PASS'
