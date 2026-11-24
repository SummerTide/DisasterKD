#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/reproduce.sh {floodnet|rescuenet} RAW_ROOT {pspnet18|deeplabv3-18} [training options]" >&2
    exit 2
fi
dataset=$1
raw_root=$2
student=$3
shift 3
case "$dataset" in floodnet|rescuenet) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
case "$student" in pspnet18|deeplabv3-18) ;; *) echo "Unknown student: $student" >&2; exit 2 ;; esac

data_root=${DISASTERKD_DATA_ROOT:-datasets}
run_root=${DISASTERKD_RUN_ROOT:-runs}
device=auto
training=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) data_root=$2; shift 2 ;;
        --run-root) run_root=$2; shift 2 ;;
        --device) device=$2; shift 2 ;;
        --no-pretrained) training+=("$1"); shift ;;
        --steps|--workers|--batch-size|--crop-size|--seed) training+=("$1" "$2"); shift 2 ;;
        *) echo "Unknown pipeline option: $1" >&2; exit 2 ;;
    esac
done
shared=(--dataset "$dataset" --data-root "$data_root" --run-root "$run_root" --device "$device")
if [[ -d "$data_root/$dataset" ]]; then
    echo "Using prepared data in $data_root/$dataset"
else
    python -m disasterkd prepare --dataset "$dataset" --raw-root "$raw_root" --data-root "$data_root"
fi
python -m disasterkd teacher "${shared[@]}" "${training[@]}"
python -m disasterkd distill "${shared[@]}" --student "$student" "${training[@]}"
python -m disasterkd evaluate "${shared[@]}" --student "$student"
