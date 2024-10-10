#!/bin/sh

env="SingleCombat"
scenario="1v1/ShootMissile/Hierarchy"
algo="ppo"
exp="v1"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0
D:\\Anaconda\\envs\\py38\\python.exe ../train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 8 --cuda --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 5000 --num-env-steps 1e6 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "256 256" --act-hidden-size "256 256" --recurrent-hidden-size 256 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-prior \
    --user-name "wangjian20001006" --use-wandb