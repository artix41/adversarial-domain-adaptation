# Adversarial Domain Adaptation

Framework to try and benchmark multiple adversarial domain adaptation algorithms (Cycle-GAN, UNIT, etc.).

In construction...

## Usage

To run an experiment, create a config file (following the templates in `configs/`) and run the command:
```bash
python run-experiments.py configs/my-config-file.yaml outputs -v 3
```

You will then be able to get the results (generated images and summaries for tensorboard) inside the folder `outputs/my-config-file/`.  
