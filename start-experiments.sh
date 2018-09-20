source activate tl-gpu
CUDA_VISIBLE_DEVICES=0 nohup python run-experiments.py configs outputs -v 3 &
