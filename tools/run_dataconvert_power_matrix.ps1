param(
    [string]$BuildDir = "cmake-build-debug",
    [string]$OutputRoot = "output\dataconvert_power_matrix_20260618",
    [string]$WslDistro = "Ubuntu-22.04",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path ".").Path
$build = Join-Path $repo $BuildDir
$exe = Join-Path $build "NMC_Simulator"
if (-not (Test-Path $exe)) {
    throw "Simulator executable not found: $exe"
}
$wslRepo = "/mnt/c/" + ($repo.Substring(3).Replace("\", "/"))
$wslBuild = "$wslRepo/$($BuildDir.Replace('\', '/'))"

$wslCheck = & wsl.exe -d $WslDistro -- bash -lc "test -x '$wslBuild/NMC_Simulator' && echo OK" 2>&1
if ($LASTEXITCODE -ne 0 -or ($wslCheck -notcontains "OK")) {
    throw "WSL distro '$WslDistro' is not available or '$wslBuild/NMC_Simulator' is not executable. wsl output: $($wslCheck -join ' ')"
}

$outRootAbs = Join-Path $repo $OutputRoot
New-Item -ItemType Directory -Force -Path $outRootAbs | Out-Null

$cases = @(
    @{ Device = "Nano"; Model = "OPT-350M"; ModelConfig = "../configs/model_config/OPT-350M.json" },
    @{ Device = "Nano"; Model = "OPT-1.3B"; ModelConfig = "../configs/model_config/OPT-1.3B.json" },
    @{ Device = "Tiny"; Model = "OPT-350M"; ModelConfig = "../configs/model_config/OPT-350M.json" },
    @{ Device = "Tiny"; Model = "OPT-1.3B"; ModelConfig = "../configs/model_config/OPT-1.3B.json" },
    @{ Device = "Small"; Model = "OPT-1.3B"; ModelConfig = "../configs/model_config/OPT-1.3B.json" },
    @{ Device = "Small"; Model = "OPT-7B"; ModelConfig = "../configs/model_config/OPT-7B.json" },
    @{ Device = "Base"; Model = "OPT-7B"; ModelConfig = "../configs/model_config/OPT-7B.json" },
    @{ Device = "Base"; Model = "OPT-13B"; ModelConfig = "../configs/model_config/opt-13B.json" }
)

$deviceConfig = @{
    Nano = @{
        Compute = "../configs/Cases/Nano/Nano_1x64x64.json"
        Dram = "../configs/Cases/Nano/Nano_2xLPDDR5.ini"
        Pim = "../configs/Cases/Nano/Nano_pim_config.json"
    }
    Tiny = @{
        Compute = "../configs/Cases/Tiny/Tiny_1x128x128.json"
        Dram = "../configs/Cases/Tiny/Tiny_4xLPDDR5.ini"
        Pim = "../configs/Cases/Tiny/Tiny_pim_config.json"
    }
    Small = @{
        Compute = "../configs/Cases/Small/Small_2x128x128.json"
        Dram = "../configs/Cases/Small/Small_4xGDDR6.ini"
        Pim = "../configs/Cases/Small/Small_pim_config.json"
    }
    Base = @{
        Compute = "../configs/Cases/Base/Base_1x256x256.json"
        Dram = "../configs/Cases/Base/Base_8xGDDR6.ini"
        Pim = "../configs/Cases/Base/Base_pim_config.json"
    }
}

$directions = @(
    @{ Name = "XPU_to_IANUS"; Scheme = "NPU" },
    @{ Name = "IANUS_to_XPU"; Scheme = "IANUS" }
)

function Read-StageSummary($path) {
    $result = @{ total_cycles = ""; pim_cycles = ""; mem_bw_util = "" }
    if (-not (Test-Path $path)) { return $result }
    $lines = Get-Content $path
    if ($lines.Count -lt 2) { return $result }
    $parts = $lines[1] -split "`t"
    $result.total_cycles = $parts[1]
    $result.pim_cycles = $parts[2]
    $result.mem_bw_util = $parts[3]
    return $result
}

function Sum-JsonField($json, [string]$field) {
    $sum = 0.0
    foreach ($prop in $json.PSObject.Properties) {
        if ($null -ne $prop.Value.$field) {
            $sum += [double]$prop.Value.$field
        }
    }
    return $sum
}

function Sum-StaticEnergy($json) {
    return (Sum-JsonField $json "background_energy") + (Sum-JsonField $json "pim_background_energy")
}

function Sum-DynamicEnergy($json) {
    $fields = @(
        "act_energy", "read_energy", "write_energy", "ref_energy", "refb_energy",
        "gwrite_energy", "comp_energy", "readres_energy", "pim_dynamic_energy"
    )
    $sum = 0.0
    foreach ($field in $fields) {
        $sum += Sum-JsonField $json $field
    }
    return $sum
}

function Parse-OpCycles($stdoutPath) {
    $ops = [ordered]@{}
    if (-not (Test-Path $stdoutPath)) { return $ops }
    foreach ($line in Get-Content $stdoutPath) {
        if ($line -match "Op ([^:]+) : ([0-9]+) cycles") {
            $ops[$matches[1]] = [int64]$matches[2]
        }
    }
    return $ops
}

function Add-PerCmdRows($rows, $case, $direction, $json) {
    $cmdPairs = @(
        @{ Cmd = "ACT"; Count = "num_act_cmds"; Energy = "act_energy" },
        @{ Cmd = "READ"; Count = "num_read_cmds"; Energy = "read_energy" },
        @{ Cmd = "WRITE"; Count = "num_write_cmds"; Energy = "write_energy" },
        @{ Cmd = "REF"; Count = "num_ref_cmds"; Energy = "ref_energy" },
        @{ Cmd = "REFB"; Count = "num_refb_cmds"; Energy = "refb_energy" },
        @{ Cmd = "GWRITE"; Count = "num_gwrite_cmds"; Energy = "gwrite_energy" },
        @{ Cmd = "COMP"; Count = "num_comp_cmds"; Energy = "comp_energy" },
        @{ Cmd = "READRES"; Count = "num_readres_cmds"; Energy = "readres_energy" },
        @{ Cmd = "PIM_DYNAMIC"; Count = "num_pim_cmds"; Energy = "pim_dynamic_energy" }
    )
    foreach ($pair in $cmdPairs) {
        $count = Sum-JsonField $json $pair.Count
        $energy = Sum-JsonField $json $pair.Energy
        $perCmd = ""
        if ($count -gt 0) {
            $perCmd = $energy / $count
        }
        $rows.Add([pscustomobject]@{
            device = $case.Device
            model = $case.Model
            direction = $direction.Name
            command = $pair.Cmd
            count = $count
            total_energy_pJ = $energy
            energy_per_command_pJ = $perCmd
        }) | Out-Null
    }
}

$summaryRows = New-Object System.Collections.Generic.List[object]
$cmdRows = New-Object System.Collections.Generic.List[object]
$opRows = New-Object System.Collections.Generic.List[object]

foreach ($case in $cases) {
    $cfg = $deviceConfig[$case.Device]
    foreach ($direction in $directions) {
        $slug = ("{0}_{1}_{2}" -f $case.Device, $case.Model, $direction.Name).ToLower().Replace(".", "p")
        $runDir = Join-Path $outRootAbs $slug
        $stdout = Join-Path $outRootAbs ("stdout_{0}.log" -f $slug)
        $simPath = Join-Path $outRootAbs ("simulation_{0}.json" -f $slug)
        $inferPath = Join-Path $outRootAbs ("inference_{0}.json" -f $slug)

        if ($SkipExisting -and (Test-Path (Join-Path $runDir "dramsim3.json"))) {
            Write-Host "Skipping existing $slug"
        }
        else {
            New-Item -ItemType Directory -Force -Path $runDir | Out-Null

            $infer = [ordered]@{
                max_batch_size = 16
                max_active_reqs = 16
                max_seq_len = 1024
                kv_cache_entry_size = 32
                allocation_scheme = $direction.Scheme
                virtual_mem_hash_enable = $false
                dram_data_container_enable = $false
                test_single_op = $false
                test_single_op_name = "dataconvert"
                test_multi_layer = $true
                test_multi_layer_name = "data_convert_weights"
                accelerate_ctrl = $false
                accelerate_method = "Loop_wise"
                dram_trace_simulation_mode = $false
                record_dram_completion_trace = $false
                gen_request = $true
                gen_request_count = 2
                gen_request_input_size = 512
                gen_request_output_size = 128
                gen_random_request = $false
                request_interval = 1
            }
            $infer | ConvertTo-Json -Depth 8 | Set-Content -Encoding ASCII $inferPath

            $sim = [ordered]@{
                compute_die_config_file_path = $cfg.Compute
                DRAM_config_file_path = $cfg.Dram
                PIM_config_file_path = $cfg.Pim
                model_config_file_path = $case.ModelConfig
                inference_config_file_path = ("../{0}" -f ($inferPath.Substring($repo.Length + 1).Replace("\", "/")))
                request_file_path = "../tests/request-traces/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv"
            }
            $sim | ConvertTo-Json -Depth 8 | Set-Content -Encoding ASCII $simPath

            Write-Host "Running $slug"
            $simRel = $simPath.Substring($repo.Length + 1).Replace("\", "/")
            $runRel = $runDir.Substring($repo.Length + 1).Replace("\", "/")
            $cmd = "cd '$wslBuild' && ./NMC_Simulator -sc '../$simRel' -o '../$runRel'"
            $elapsed = Measure-Command {
                & wsl.exe -d $WslDistro -- bash -lc $cmd *> $stdout
            }
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed for $slug. See $stdout"
            }
            Add-Content -Encoding ASCII -Path $stdout -Value ("REAL_SECONDS {0:N2}" -f $elapsed.TotalSeconds)
        }

        $jsonPath = Join-Path $runDir "dramsim3.json"
        $stagePath = Join-Path $runDir "_summary.tsv"
        if (-not (Test-Path $jsonPath)) {
            $summaryRows.Add([pscustomobject]@{
                device = $case.Device; model = $case.Model; direction = $direction.Name; status = "missing_dramsim3_json"
                latency_cycles = ""; dram_cycles = ""; runtime_seconds = ""; static_energy_pJ = ""; dynamic_energy_pJ = ""; total_energy_pJ = ""
                static_pJ_per_cycle = ""; dynamic_pJ_per_cycle = ""; total_pJ_per_cycle = ""; average_power_sum = ""; scheduler_mem_bw_util = ""
            }) | Out-Null
            continue
        }

        $json = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $stage = Read-StageSummary $stagePath
        $dramCycles = Sum-JsonField $json "num_cycles"
        $channels = @($json.PSObject.Properties).Count
        $dramCyclesPerChannel = ""
        if ($channels -gt 0) { $dramCyclesPerChannel = $dramCycles / $channels }
        $staticEnergy = Sum-StaticEnergy $json
        $dynamicEnergy = Sum-DynamicEnergy $json
        $totalEnergy = Sum-JsonField $json "total_energy"
        $avgPower = Sum-JsonField $json "average_power"

        $runtimeSeconds = ""
        if (Test-Path $stdout) {
            $realLine = (Select-String -Path $stdout -Pattern "REAL_SECONDS" | Select-Object -Last 1).Line
            if ($realLine -match "REAL_SECONDS\s+([0-9.]+)") { $runtimeSeconds = $matches[1] }
        }

        $summaryRows.Add([pscustomobject]@{
            device = $case.Device
            model = $case.Model
            direction = $direction.Name
            status = "completed"
            latency_cycles = $stage.total_cycles
            dram_cycles_per_channel = $dramCyclesPerChannel
            runtime_seconds = $runtimeSeconds
            static_energy_pJ = $staticEnergy
            dynamic_energy_pJ = $dynamicEnergy
            total_energy_pJ = $totalEnergy
            static_pJ_per_cycle = if ($dramCycles -gt 0) { $staticEnergy / $dramCycles } else { "" }
            dynamic_pJ_per_cycle = if ($dramCycles -gt 0) { $dynamicEnergy / $dramCycles } else { "" }
            total_pJ_per_cycle = if ($dramCycles -gt 0) { $totalEnergy / $dramCycles } else { "" }
            average_power_sum = $avgPower
            scheduler_mem_bw_util = $stage.mem_bw_util
        }) | Out-Null

        Add-PerCmdRows $cmdRows $case $direction $json
        $ops = Parse-OpCycles $stdout
        foreach ($entry in $ops.GetEnumerator()) {
            $opRows.Add([pscustomobject]@{
                device = $case.Device
                model = $case.Model
                direction = $direction.Name
                operation = $entry.Key
                cycles = $entry.Value
            }) | Out-Null
        }
    }
}

$summaryRows | Export-Csv -NoTypeInformation -Encoding ASCII (Join-Path $outRootAbs "summary.csv")
$cmdRows | Export-Csv -NoTypeInformation -Encoding ASCII (Join-Path $outRootAbs "per_command_energy.csv")
$opRows | Export-Csv -NoTypeInformation -Encoding ASCII (Join-Path $outRootAbs "per_operation_latency.csv")

Write-Host "Wrote:"
Write-Host (Join-Path $outRootAbs "summary.csv")
Write-Host (Join-Path $outRootAbs "per_command_energy.csv")
Write-Host (Join-Path $outRootAbs "per_operation_latency.csv")
