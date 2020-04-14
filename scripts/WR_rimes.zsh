echo "running: nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 rimes /data/mspringe/rimes/gt/ /data/mspringe/rimes/imgs/ --gpu=$2 --estimator=$3 --name=$4 ${@:5}"
nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 rimes /data/mspringe/rimes/gt/ /data/mspringe/rimes/imgs/ --gpu=$2 --estimator=$3 --name=$4 ${@:5}

