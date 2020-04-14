echo "\n\nrunning: nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ hws /data/mspringe/hw_synth/groundtruth/ /vol/corpora/document-image-analysis/hw-synth/Images_90K_Normalized/ --max_iter=$1 --name=$2 --gpu_idx=$3 ${@:4}\n\n"
nice python3 ../src/training/phocnet_trainer.py /data/mspringe/PHOCNet_models/ hws /data/mspringe/hw_synth/groundtruth/ /vol/corpora/document-image-analysis/hw-synth/Images_90K_Normalized/ --max_iter=$1 --name=$2 --gpu_idx=$3 ${@:4}

