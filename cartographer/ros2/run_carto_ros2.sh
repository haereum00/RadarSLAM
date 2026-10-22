#!/bin/bash

set -e
cd "$(dirname "$0")/../.."
CFG=${1:-configs/rig2.yaml}
HERE=cartographer/ros2
PY=.venv/bin/python
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM="$HOME/.local/bin/micromamba run -n ros_carto"
ENV=$MAMBA_ROOT_PREFIX/envs/ros_carto

OUT=$($PY -c "from radarize.config import cfg; cfg.merge_from_file('$CFG'); print(cfg.OUTPUT_DIR.rstrip('/'))")
TEST=$($PY -c "from radarize.config import cfg; cfg.merge_from_file('$CFG'); print(' '.join(cfg.DATASET.TEST_SPLIT))")
C=$(realpath -m $OUT)/carto
mkdir -p $C/input $C/output

PROTO=$HERE/proto_py
if [ ! -d $PROTO/cartographer ]; then
    SRC=$HOME/.cache/cartographer_src
    [ -d $SRC ] || git clone -q --depth 1 --branch 2.0.0 https://github.com/cartographer-project/cartographer.git $SRC
    mkdir -p $PROTO
    (cd $SRC && $ENV/bin/protoc -I. --python_out=$OLDPWD/$PROTO $(find cartographer -name "*.proto"))
fi

for s in $TEST; do
    echo "== $s"
    $PY tools/export_cartographer.py --cfg=$CFG --odom_path=$OUT/odometry/$s.txt \
        --scan_path=$OUT/unet/$s.npz --output_path=$C/input/$s.bag > /dev/null
    rm -rf $C/input/${s}_ros2
    $MM python $HERE/ros1_to_humble.py $C/input/$s.bag $C/input/${s}_ros2

    $MM $ENV/lib/cartographer_ros/cartographer_offline_node \
        -configuration_directory $PWD/$HERE -configuration_basenames radar.lua \
        -urdf_filenames $PWD/$HERE/radar_frame.urdf \
        -bag_filenames $C/input/${s}_ros2 -save_state_filename $C/output/$s.pbstream > $C/output/$s.log 2>&1

    $MM python $HERE/pbstream_to_tum.py $C/output/$s.pbstream $C/output/$s.txt $PROTO
    $MM $ENV/lib/cartographer_ros/cartographer_pbstream_to_ros_map \
        -pbstream_filename $C/output/$s.pbstream -map_filestem $C/output/${s}_map -resolution 0.05 > /dev/null 2>&1

    $PY $HERE/eval_slam.py $s $C/output $OUT
done
