printf "\n\nrunning: nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 iam /data/mspringe/iamDB/ascii/ /data/mspringe/iamDB/forms/png/ --gpu=$2 --estimator=$3 --name=$4\n"
nice python3 ../src/experiments/word_rec.py /data/mspringe/experiments/WR/ $1 iam /data/mspringe/iamDB/ascii/ /data/mspringe/iamDB/forms/png/ --gpu=$2 --estimator=$3 --name=$4
