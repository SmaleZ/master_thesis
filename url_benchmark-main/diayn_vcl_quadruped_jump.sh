#!/bin/sh
# This is a comment!


for snapshot_ts in 100000 500000 1000000 2000000
do
    python finetune.py agent=diayn_vcl task=quadruped_jump snapshot_ts=$snapshot_ts obs_type=states agent=diayn_vcl reward_free=false seed=4
done
