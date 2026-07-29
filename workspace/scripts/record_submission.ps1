[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Concat', 'Greater', 'IndexAdd', 'Transpose', 'SquareSumV1')]
    [string]$Operator,

    [Parameter(Mandatory = $true)]
    [string]$ZipPath,

    [string]$PlatformScore = '',
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$resolvedZip = (Resolve-Path -LiteralPath $ZipPath).Path
$hash = (Get-FileHash -LiteralPath $resolvedZip -Algorithm SHA256).Hash.ToLowerInvariant()
$commit = git -C $Root rev-parse HEAD
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
$size = (Get-Item -LiteralPath $resolvedZip).Length
$logPath = Join-Path $Root "raceS9\$Operator\notes\submissions.md"

if (-not (Test-Path -LiteralPath $logPath)) {
    @(
        '# 提交记录',
        '',
        '| 时间 | Git commit | ZIP | 字节 | SHA-256 | 平台成绩 |',
        '|---|---|---|---:|---|---|'
    ) | Set-Content -LiteralPath $logPath -Encoding utf8
}

$zipName = Split-Path -Leaf $resolvedZip
$row = "| $timestamp | ``$commit`` | ``$zipName`` | $size | ``$hash`` | $PlatformScore |"
Add-Content -LiteralPath $logPath -Value $row -Encoding utf8

Write-Output "Recorded submission in $logPath"
Write-Output "SHA-256: $hash"
