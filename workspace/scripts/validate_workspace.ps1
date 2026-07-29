[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$operators = @('Concat', 'Greater', 'IndexAdd', 'Transpose', 'SquareSumV1')
$missing = [System.Collections.Generic.List[string]]::new()

$requiredRootPaths = @(
    'README.md',
    'docs\WORKFLOW.md',
    'docs\SUBMISSION_CHECKLIST.md',
    'utils\repos\README.md',
    'utils\skills\README.md',
    'raceS9\README.md'
)

foreach ($relativePath in $requiredRootPaths) {
    $path = Join-Path $Root $relativePath
    if (-not (Test-Path -LiteralPath $path)) {
        $missing.Add($relativePath)
    }
}

foreach ($operator in $operators) {
    foreach ($relativePath in @('README.md', 'op_host', 'op_kernel', 'tests\official')) {
        $path = Join-Path $Root (Join-Path "raceS9\$operator" $relativePath)
        if (-not (Test-Path -LiteralPath $path)) {
            $missing.Add("raceS9\$operator\$relativePath")
        }
    }

    $officialTest = Join-Path $Root "raceS9\$operator\tests\official\test_op.py"
    if (-not (Test-Path -LiteralPath $officialTest)) {
        $missing.Add("raceS9\$operator\tests\official\test_op.py")
    }
}

$repoRoot = Join-Path $Root 'utils\repos'
$repos = @('cannbot-skills', 'cann-learning-hub', 'ascend-samples-operator', 'cann-ops', 'asc-tools')
$repoRows = foreach ($repo in $repos) {
    $path = Join-Path $repoRoot $repo
    if (-not (Test-Path -LiteralPath (Join-Path $path '.git'))) {
        $missing.Add("utils\repos\$repo\.git")
        continue
    }

    [pscustomobject]@{
        Repository = $repo
        Commit = (git -C $path rev-parse --short=12 HEAD)
        Branch = (git -C $path branch --show-current)
    }
}

if ($missing.Count -gt 0) {
    Write-Error ("Workspace validation failed. Missing:`n- " + ($missing -join "`n- "))
    exit 1
}

Write-Output 'Workspace structure: OK'
$repoRows | Format-Table -AutoSize
exit 0
