<#
.SYNOPSIS
    Pull every V4 figure (*.png) and LaTeX table (*.tex) from the cluster to the
    local OneDrive folder, keeping the remote directory structure.

.DESCRIPTION
    Run this ON WINDOWS (PowerShell 5.1 or 7+).  It

      1. asks the cluster to tar up *.png / *.tex from the transfer_v4_final tree
         (scripts/pack_v4_figures.sh, which skips the _merged_* symlink trees),
      2. scp's that one tarball down (one transfer instead of ~225 round trips),
      3. unpacks it into -DestDir, recreating best/ btl/ bbs/ overlay_*/ venn_*/,
      4. deletes both temporary tarballs.

    Needs ssh + scp + tar, all shipped with Windows 10/11 (OpenSSH client + bsdtar).
    Set up key-based ssh to the cluster first, or you will be prompted twice.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\sync_v4_figures.ps1

.EXAMPLE
    # a different remote tree, and overwrite nothing that is already local
    .\sync_v4_figures.ps1 -RemoteDir /proj/.../transfer_v4_final2 -SkipExisting
#>
[CmdletBinding()]
param(
    # ssh target; must be reachable as `ssh <RemoteHost>` (use a ~/.ssh/config alias
    # if you go through a jump host).
    [string]$RemoteHost = "colglx1200",

    [string]$RemoteDir  = "/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4_final",

    # The repo checkout on the cluster, for scripts/pack_v4_figures.sh.
    [string]$RemoteRepo = "/proj/gpu_mtk53728/projects/UQ",

    # Local destination.  A subfolder is used so the six mode/overlay folders do not
    # land loose in 4_UQ; point it straight at 4_UQ if you would rather they did.
    [string]$DestDir = "$env:USERPROFILE\OneDrive - MediaTek\Documents\1_Project\4_UQ\transfer_v4_final",

    # Do not overwrite files that already exist locally.
    [switch]$SkipExisting,

    # List what would be transferred, then stop.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Require-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "$name not found on PATH. On Windows 10/11 install the OpenSSH client " +
              "(Settings > Apps > Optional Features) -- tar ships with the OS."
    }
}
Require-Tool ssh
Require-Tool scp
Require-Tool tar

if ($DryRun) {
    Write-Host "Files that would be pulled from ${RemoteHost}:${RemoteDir}:" -ForegroundColor Cyan
    ssh $RemoteHost "cd '$RemoteDir' && find . \( -name '*.png' -o -name '*.tex' \) -not -path '*/_merged_*' | sort"
    Write-Host "`nDestination: $DestDir" -ForegroundColor Cyan
    return
}

Write-Host "[1/4] packing on $RemoteHost ..." -ForegroundColor Cyan
# The pack script prints progress on stderr and the tarball path as its last stdout
# line, so this captures the path only.
$remoteTgz = (ssh $RemoteHost "bash '$RemoteRepo/scripts/pack_v4_figures.sh' '$RemoteDir'" |
              Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $remoteTgz) { throw "remote packing failed" }
Write-Host "      $remoteTgz"

$localTgz = Join-Path $env:TEMP (Split-Path $remoteTgz -Leaf)
try {
    Write-Host "[2/4] downloading ..." -ForegroundColor Cyan
    scp -q "${RemoteHost}:$remoteTgz" $localTgz
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }

    Write-Host "[3/4] unpacking into $DestDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    # -k keeps existing files; without it tar overwrites, which is what we want by
    # default so a re-run picks up regenerated figures.
    $tarArgs = @("-xzf", $localTgz, "-C", $DestDir)
    if ($SkipExisting) { $tarArgs = @("-xzkf", $localTgz, "-C", $DestDir) }
    & tar @tarArgs
    if ($LASTEXITCODE -ne 0 -and -not $SkipExisting) { throw "tar extraction failed" }

    $n = (Get-ChildItem $DestDir -Recurse -Include *.png, *.tex).Count
    Write-Host "[4/4] done: $n png/tex files under $DestDir" -ForegroundColor Green
}
finally {
    Remove-Item $localTgz -ErrorAction SilentlyContinue
    ssh $RemoteHost "rm -f '$remoteTgz'" 2>$null
}
