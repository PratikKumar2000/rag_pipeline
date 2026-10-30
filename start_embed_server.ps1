param(
    [string]$Model = "nomic-embed-text",
    [int]$Port = 8081,
    [int]$Parallel = 4,
    [int]$SlotContext = 2048
)

$lib = Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama"
$from = ollama show $Model --modelfile | Select-String '^FROM\s+(.+)$'
if (-not $from) { throw "Model '$Model' is not installed in Ollama." }
$blob = $from.Matches[0].Groups[1].Value.Trim()

# Ollama's bundled llama-server only finds the GPU when pointed at the Vulkan backend explicitly.
$env:GGML_BACKEND_PATH = Join-Path $lib "vulkan\ggml-vulkan.dll"
$env:PATH = "$lib\vulkan;$lib;$env:PATH"

# Embedding needs a whole chunk in one micro-batch; sizing it to all slots lets every slot run in one pass.
$total = $SlotContext * $Parallel
& (Join-Path $lib "llama-server.exe") `
    --model $blob --host 127.0.0.1 --port $Port --no-webui `
    --embedding -np $Parallel -c $total -b $total -ub $total
