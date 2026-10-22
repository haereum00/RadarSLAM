#!/bin/bash

set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
CFG=configs/rig2.yaml

SEQS=$($PY -c "
from radarize.config import cfg; cfg.merge_from_file('$CFG')
d = cfg.DATASET; print(' '.join(sorted(set(d.TRAIN_SPLIT + d.VAL_SPLIT + d.TEST_SPLIT))))")
TEST=$($PY -c "
from radarize.config import cfg; cfg.merge_from_file('$CFG'); print(' '.join(cfg.DATASET.TEST_SPLIT))")

if [ "$1" != "train" ]; then
    for s in $SEQS; do
        [ -f data/$s.npz ]    || $PY tools/create_dataset.py    --cfg=$CFG --bag_path=data/$s.bag
        [ -f data/${s}_3d.npz ] || $PY tools/create_dataset_3d.py --cfg=$CFG --bag_path=data/$s.bag
    done
fi

$PY tools/train_flow.py --cfg=$CFG
$PY tools/test_flow.py  --cfg=$CFG
$PY tools/train_rot.py  --cfg=$CFG
$PY tools/test_rot.py   --cfg=$CFG
for s in $TEST; do $PY tools/test_odom.py --cfg=$CFG --npz_path=data/$s.npz; done
$PY tools/train_unet.py --cfg=$CFG
for s in $TEST; do $PY tools/test_unet.py --cfg=$CFG --npz_path=data/$s.npz; done

$PY tools/train_unet3d.py --cfg=$CFG
for s in $TEST; do $PY tools/test_unet3d.py --cfg=$CFG --npz_path=data/$s.npz; done

cartographer/ros2/run_carto_ros2.sh $CFG

OUT=$($PY -c "from radarize.config import cfg; cfg.merge_from_file('$CFG'); print(cfg.OUTPUT_DIR.rstrip('/'))")
for s in $TEST; do
    $PY tools/map3d.py --cfg=$CFG --seq=$s --poses=gt
    $PY tools/map3d.py --cfg=$CFG --seq=$s --poses=$OUT/carto/output/$s.txt
done
