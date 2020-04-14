echo "\n\nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ rimes /data/mspringe/rimes/gt/ /data/mspringe/rimes/imgs/ --max_iter=$1 --name=$2 --gpu_idx=$3 ${@:4}\n"
nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ rimes /data/mspringe/rimes/gt/ /data/mspringe/rimes/imgs/ --max_iter=$1 --name=$2 --gpu_idx=$3 ${@:4}

