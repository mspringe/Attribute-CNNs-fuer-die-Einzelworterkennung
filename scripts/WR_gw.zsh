echo "running: nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 gw /data/mspringe/gwdb/almazan/queries/queries.gtp /data/mspringe/gwdb/almazan/images --gpu=$2 --estimator=$3 --name=$4 --k_fold=$5 ${@:6}"
nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 gw /data/mspringe/gwdb/almazan/queries/queries.gtp /data/mspringe/gwdb/almazan/images --gpu=$2 --estimator=$3 --name=$4 --k_fold=$5 ${@:6}

