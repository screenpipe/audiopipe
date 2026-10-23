# screenpipe — AI that knows everything you've seen, said, or heard
# https://screenpipe.com
# Manual native validation. Run only on a disposable Windows dev VM.
param([string] $ResultDir = 'work\bark-windows-validation')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ([Environment]::OSVersion.Platform -ne 'Win32NT') { throw 'This validation requires native Windows' }
$repo = (git rev-parse --show-toplevel).Trim()
Set-Location $repo
$ResultDir = if ([IO.Path]::IsPathRooted($ResultDir)) {
  [IO.Path]::GetFullPath($ResultDir)
} else {
  [IO.Path]::GetFullPath((Join-Path $repo $ResultDir))
}
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$previousBarkModelPath = $env:BARK_MODEL_PATH
$result = [ordered]@{ state = 'running'; testedSha = (git rev-parse HEAD).Trim(); stages = @(); hardwareGpuBenchmark = $false }
function Run-Native([string] $Name, [string] $Program, [string[]] $Arguments, [bool] $AllowFailure = $false) {
  $exe = (Get-Command $Program -ErrorAction Stop).Source
  $out = Join-Path $ResultDir "$Name.stdout.log"
  $err = Join-Path $ResultDir "$Name.stderr.log"
  # Start-Process -Wait follows the launched process tree on Windows. Cargo can
  # leave unrelated descendants holding inherited handles, which makes the
  # validator hang even after cargo and its test binaries have exited. Wait on
  # the native process itself so its exit code remains authoritative.
  $start = @{ FilePath = $exe; WorkingDirectory = $repo; RedirectStandardOutput = $out; RedirectStandardError = $err; PassThru = $true; NoNewWindow = $true }
  if ($Arguments.Count -gt 0) { $start.ArgumentList = $Arguments }
  $p = Start-Process @start
  # Force .NET to retain the native process handle. Without this, long-running
  # processes can finish before PowerShell opens it and ExitCode is then null.
  [void] $p.Handle
  $p.WaitForExit()
  $script:result.stages += [ordered]@{ name = $Name; exitCode = $p.ExitCode; stdout = [IO.Path]::GetFileName($out); stderr = [IO.Path]::GetFileName($err) }
  if ($p.ExitCode -ne 0 -and -not $AllowFailure) { throw "$Name failed with native exit code $($p.ExitCode)" }
  return [int] $p.ExitCode
}
try {
  $result.rustc = (& rustc --version | Out-String).Trim()
  $result.os = [Environment]::OSVersion.VersionString
  $result.videoControllers = @(Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID, DriverVersion)
  $result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $ResultDir 'result.json')
  $flags = @('--no-default-features', '--features', 'bark-onnx,ort-defaults,directml', '--release')
  Run-Native 'unit-tests' 'cargo.exe' (@('test') + $flags + @('--lib', '--test', 'bark_weights', '--', '--test-threads=1')) | Out-Null
  Run-Native 'build' 'cargo.exe' (@('build') + $flags + @('--example', 'download_bark', '--example', 'bark_onnx_bench')) | Out-Null
  $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $repo 'target' }
  $download = Join-Path $target 'release\examples\download_bark.exe'
  $bench = Join-Path $target 'release\examples\bark_onnx_bench.exe'
  Run-Native 'download' $download @() | Out-Null
  $downloadResult = Get-Content (Join-Path $ResultDir 'download.stdout.log') -Raw | ConvertFrom-Json
  $env:BARK_MODEL_PATH = $downloadResult.model_path
  Run-Native 'model-digests' 'cargo.exe' (@('test') + $flags + @('--test', 'bark_weights', '--', '--ignored')) | Out-Null
  Run-Native 'prepare-fixtures' 'bash.exe' @('bench/bark/prepare.sh') | Out-Null
  Run-Native 'cpu' $bench @('bark', 'bench/bark/fixtures.json', '1', 'cpu') | Out-Null
  Run-Native 'directml-or-cpu' $bench @('bark', 'bench/bark/fixtures.json', '1', 'directml-or-cpu') | Out-Null
  $strictExit = Run-Native 'directml-strict' $bench @('bark', 'bench/bark/fixtures.json', '1', 'directml') $true
  $cpu = Get-Content (Join-Path $ResultDir 'cpu.stdout.log') -Raw | ConvertFrom-Json
  $automatic = Get-Content (Join-Path $ResultDir 'directml-or-cpu.stdout.log') -Raw | ConvertFrom-Json
  if ($cpu.execution_provider -ne 'CPU') { throw 'Explicit CPU request did not report CPU' }
  foreach ($report in @($cpu, $automatic)) {
    if (@($report.cases).Count -ne 5) { throw 'Expected five real audio fixtures' }
    $meeting = @($report.cases | Where-Object name -EQ 'accuracy3')[0].transcript
    if ($meeting -notmatch '(?i)screen\s*pipe' -or $meeting -notmatch '(?i)meeting') { throw 'Real speech did not contain the expected fixture terms' }
  }
  $result.cpuPassed = $true
  $result.automaticExecutionProvider = $automatic.execution_provider
  $result.directmlStrictExitCode = $strictExit
  $result.directmlSessionPassed = $strictExit -eq 0
  # DirectML may execute on a software adapter. Session registration alone is
  # deliberately never labeled a physical-GPU performance measurement.
  $result.state = 'passed'
} catch {
  $result.state = 'failed'
  $result.failure = $_.Exception.Message
  throw
} finally {
  $result.completedAt = [DateTime]::UtcNow.ToString('o')
  $result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $ResultDir 'result.json')
  if ($null -ne $previousBarkModelPath) { $env:BARK_MODEL_PATH = $previousBarkModelPath } else { Remove-Item Env:\BARK_MODEL_PATH -ErrorAction SilentlyContinue }
}
