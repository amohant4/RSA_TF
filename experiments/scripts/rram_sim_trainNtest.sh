#!/bin/bash
# Usage:
# ./experiments.scripts/rram_sim_trainNtest.sh <network> <dataset> <write to pickle>
# ./experiments/scripts/rram_sim_trainNtest.sh lenet mnist 0  

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=$1
DATASET=$2
WRITE_TO_PICKLE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $NET_NAME in
  mlp_1)
	NET="mlp_1"
	;;
  mlp_2)
	NET="mlp_2"
	;;
  lenet)
	NET="leNet"
	;;
  lenet_sram)
	NET="leNet_sram"
	;;
  cifarcnn)
	NET="cifarCNN"
	;;
  cifarcnn_sram)
	NET="cifarCNN_sram"  
	;;
  alenet)
	NET="alexNet"	  
	;;
  *)
	echo "No network architecture given"
	exit
	;;		
esac

case $DATASET in
  mnist)
    IMDB="mnist"
    ITERS=50000
    ;;
  cifar10)
    IMDB="cifar10"
    ITERS=150000
    ;;	    
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/rramSimulator_${NET_NAME}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train.py \
  --dataset ${IMDB} \
  --iters ${ITERS} \
  --network ${NET} \
  --write_pickle ${WRITE_TO_PICKLE} \
  ${EXTRA_ARGS}

#set +x
#NET_FINAL=`grep -B 1 "Done Training" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
#set -x
#
#time python ./tools/test.py --device ${DEV} --device_id ${DEV_ID} \
#  --data_dir ${TRAIN_IMDB} \
#  --network ${NET} \
#  --weights ${NET_FINAL} \
#  ${EXTRA_ARGS}
