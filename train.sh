CUDA_VISIBLE_DEVICES=1 python main.py --no_vols 70 --test_vols 30 --loss 0.7*L1+0.3*TV --lr 0.002 --encoder conv --lr_decay 15 --run_name relu
