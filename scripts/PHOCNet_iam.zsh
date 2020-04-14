echo "\n\nrunning: nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ iam /data/mspringe/iamDB/ascii/ /data/mspringe/iamDB/forms/png/ --max_iter=$1 --name=$2 --gpu_idx=$3 ${@:4}\n"
nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ iam /data/mspringe/iamDB/ascii/ /data/mspringe/iamDB/forms/png/ --max_iter=$1 --name=$2 --gpu_idx=$3 ${@:4}

