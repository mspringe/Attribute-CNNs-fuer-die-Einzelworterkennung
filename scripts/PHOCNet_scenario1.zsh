printf "\n\nrunning: nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ iam /data/mspringe/iamDB/ascii/words.txt /data/mspringe/iamDB/forms/png/ --max_iter=$1 --name=$2 --gpu_idx=$3\n"
nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ iam /data/mspringe/iamDB/ascii/words.txt /data/mspringe/iamDB/forms/png/ --max_iter=$1 --name=$2 --gpu_idx=$3
