#!/bin/bash

source $HOME/.local/miniconda3/etc/profile.d/conda.sh
conda activate drl-sampler

ENV_NAME=Pong-v4
NUM_COLLECT_PER_ENV=500

for NUM_ENV in 16 32 64 1 2 4 8
do
    NUM_COLLECT=$(($NUM_COLLECT_PER_ENV*$NUM_ENV))
    python compare_same_config.py --env=$ENV_NAME --num-envs=$NUM_ENV --num-collect=$NUM_COLLECT  --architecture=subproc
    python compare_same_config.py --env=$ENV_NAME --num-envs=$NUM_ENV --num-collect=$NUM_COLLECT  --architecture=subproc --run-auturi

     
    for ((var=1 ; var <= $NUM_ENV ; var*=2));
    do
        python compare_same_config.py --env=$ENV_NAME --num-envs=$NUM_ENV --num-collect=$NUM_COLLECT  --architecture=rllib --num-actors=$var 
        python compare_same_config.py --env=$ENV_NAME --num-envs=$NUM_ENV --num-collect=$NUM_COLLECT  --architecture=rllib --num-actors=$var  --run-auturi

    done
 



  #echo $NUM_COLLECT
done


# 