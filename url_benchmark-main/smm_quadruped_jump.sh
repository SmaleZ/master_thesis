#!/bin/sh
# This is a comment!

# python finetune.py agent=diayn_vcl task=quadruped_jump snapshot_ts=2000000 obs_type=states agent=diayn_vcl reward_free=false seed=4

for seed in 3 4
do
    for snapshot_ts in 100000 500000 1000000 2000000
    do
        python finetune.py agent=smm task=quadruped_jump snapshot_ts=$snapshot_ts obs_type=states agent=smm reward_free=false seed=$seed
    done
done
