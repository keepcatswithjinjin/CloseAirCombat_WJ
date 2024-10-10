#!/bin/sh
env="SingleControl"
scenario="1/heading"
algo="ppo"
exp="v1"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0
 D:\\Anaconda\\envs\\py38\\python.exe ../render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 4 --n-rollout-threads 8 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 2 --buffer-size 5000 --num-env-steps 1e6 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "256 256" --act-hidden-size "256 256" --recurrent-hidden-size 256 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --model-dir '../results/SingleControl/1/heading/ppo/v1/wandb/run-20240928_192106-adjrffq9/files'
