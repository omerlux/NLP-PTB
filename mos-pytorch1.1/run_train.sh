python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 450 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB-20201018-170341 --single_gpu --gpu_device 4 --continue_train

python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 450 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB_TRAINMC_DIV --mc_eval 15 --mc_freq 10 --single_gpu --gpu_device 5