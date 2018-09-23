source activate tl-gpu
rm nohup.out
nohup python run-experiments.py configs/$1.yaml outputs -v 3 &
