source activate tl-gpu
CUDA_VISIBLE_DEVICES=0 python run-experiments.py configs outputs -v 3
