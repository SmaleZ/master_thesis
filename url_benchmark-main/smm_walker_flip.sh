#!/bin/sh
# This is a comment!

for seed in 2 3
do
    for snapshot_ts in 100000 500000 1000000 2000000
    do
        python finetune.py agent=smm task=walker_flip snapshot_ts=$snapshot_ts obs_type=states agent=smm reward_free=false seed=$seed
    done
done