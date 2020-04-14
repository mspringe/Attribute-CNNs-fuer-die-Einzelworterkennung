echo "running: nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 hws /data/mspringe/hw_synth/groundtruth/ /vol/corpora/document-image-analysis/hw-synth/Images_90K_Normalized/ --gpu=$2 --estimator=$3 --name=$4 ${@:5}"
nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 hws /data/mspringe/hw_synth/groundtruth/ /vol/corpora/document-image-analysis/hw-synth/Images_90K_Normalized/ --gpu=$2 --estimator=$3 --name=$4 ${@:5}

