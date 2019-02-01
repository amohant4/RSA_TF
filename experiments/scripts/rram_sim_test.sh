#!/bin/bash
# Usage:
# ./experiments/scripts/rram_sim_trainNtest.sh <device> <device_id> <network> <dataset>
# ./experiments/scripts/rram_sim_trainNtest.sh cpu 0 mlp_2 mnist  

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=$1
DEV_ID=$2
NET_NAME=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $NET_NAME in
  mlp_1)
	TRAIN_NET="mlp_1_train"
	TEST_NET="mlp_1_test"
	;;
  mlp_2)
	TRAIN_NET="mlp_2_train"
	TEST_NET="mlp_2_test"
	;;
  leNet_org)
	TRAIN_NET="leNet_org"
	TEST_NET="leNet_org"
	;;
  *)
	echo "No network architecture given"
	exit
	;;		
esac


case $DATASET in
  mnist)
    TRAIN_IMDB="../data/MNIST/"
    TEST_IMDB="../data/MNIT/"
    ITERS=20000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

set +x
NET_FINAL="output/retrained_leNet_org_iter_50000.ckpt"
set -x

time python ./tools/test.py --device ${DEV} --device_id ${DEV_ID} \
  --data_dir ${TRAIN_IMDB} \
  --network ${TEST_NET} \
  --weights ${NET_FINAL} \
  ${EXTRA_ARGS}
