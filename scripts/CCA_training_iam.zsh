#!/usr/bin/env bash
echo "nice python3 ../src/training/cca_cross_validation.py /data/mspringe/CCA/ iam /data/mspringe/iamDB/ascii/ /data/mspringe/iamDB/forms/png/ $1 --gpu_idx=$2 --name=$3 ${@:4}"
nice python3 ../src/training/cca_cross_validation.py /data/mspringe/CCA/ iam /data/mspringe/iamDB/ascii/ /data/mspringe/iamDB/forms/png/ $1 --gpu_idx=$2 --name=$3 ${@:4}

