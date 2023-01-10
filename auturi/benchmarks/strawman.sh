#!/bin/bash

ENV_LIST="
BipedalWalkerHardcore-v3
MinitaurBulletEnv-v0
academy_counterattack_hard
academy_3_vs_1_with_keeper
academy_run_pass_and_shoot_with_keeper
stock
portfolio
merge
bottleneck
"


for ENV in $ENV_LIST;do
    python /workspace/auturi/auturi/benchmarks/run_strawman.py --env=$ENV --tuner-mode=maximum --cuda;
    sleep 10;
    python /workspace/auturi/auturi/benchmarks/run_strawman.py --env=$ENV --tuner-mode=maximum;
    sleep 10;

done

