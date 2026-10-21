# Shared by the queue scripts (sourced, not executed): environment, model/dataset resolution, task arguments, job runner.
# Environment overrides:
#   PY=<python>             interpreter (default .venv/bin/python if present, else python)
#   MODEL=<id|path>         checkpoint (default GSAI-ML/LLaDA-8B-Instruct or Dream-org/Dream-v0-Instruct-7B)
#   MODEL_LABEL=<tag>       model tag in result names (default: the family, llada | dream)
#   DATASET=<path|id>       dataset override for every job; DATA_DIR=<dir> uses <dir>/<task>.arrow when it exists
#   GEN=<n> BLOCK=<n>       generation / block length (default 256 / 32)
#   LIMIT=<n>               evaluate only n problems (smoke test; result prefix becomes smoke_<prefix>)
#   PREFIX=<name>           result-name prefix; OFFLINE=1 uses cached model files only; DRY=1 prints the commands
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
export TOKENIZERS_PARALLELISM=false
[ -n "$OFFLINE" ] && export HF_HUB_OFFLINE=1
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
gen=${GEN:-256}; block=${BLOCK:-32}; limit=${LIMIT:-}
mkdir -p logs results configs

set_prefix() { prefix=${PREFIX:-$([ -z "$limit" ] && echo $1 || echo smoke_$1)}; }

resolve_model() {
  family=$1
  case $family in
    llada) MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct};;
    dream) MODEL=${MODEL:-Dream-org/Dream-v0-Instruct-7B};;
    *) echo "unknown family '$family' (llada|dream)"; exit 2;;
  esac
  label=${MODEL_LABEL:-$family}
}

task_args() {
  case $1 in
    gsm8k)     script=eval_gsm8k_hf.py; extra="--num-samples ${limit:-0}";;
    humaneval) script=eval_humaneval_hf.py; extra=${limit:+--limit $limit};;
    mbpp|math500|ifeval) script=eval_bench_hf.py; extra="--task $1 ${limit:+--limit $limit}";;
    *) echo "unknown task '$1' (gsm8k|humaneval|mbpp|math500|ifeval)"; exit 2;;
  esac
  local data=${DATASET:-}
  [ -z "$data" ] && [ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/$1.arrow" ] && data=$DATA_DIR/$1.arrow
  [ -n "$data" ] && extra="$extra --dataset $data"
}

run_job() {
  local gpu=$1 job=$2 task=$3 out=$4 cfg=$5; shift 5
  local queue=logs/${prefix}_${label}_gpu${gpu}.log
  task_args $task
  local cmd="$PY -u $script --model-family $family --model $MODEL ${OFFLINE:+--local-files-only} --output $out --gen-length $gen --block-length $block --steps $gen $extra ${cfg:+--variant-config $cfg} $*"
  [ -n "$DRY" ] && { echo "CUDA_VISIBLE_DEVICES=$gpu $cmd"; return; }
  echo "=== $(date) START $job -> $out" >> $queue
  CUDA_VISIBLE_DEVICES=$gpu $cmd > logs/$(basename $out).out 2>&1
  local rc=$?
  echo "=== $(date) END $job exit=$rc" >> $queue
  [ $rc -ne 0 ] && { echo "=== $(date) QUEUE STOPPED gpu$gpu after $job failed" >> $queue; exit $rc; }
}

finish_queue() { [ -z "$DRY" ] && echo "=== $(date) QUEUE DONE gpu$1" >> logs/${prefix}_${label}_gpu$1.log; }
