# M-Layer nanoGPT
nanoGPT but with MLPs replaced by M-layers: https://arxiv.org/pdf/2008.03936.
```
mlayer:
step 2000: train loss 1.5898, val loss 1.7110
vs baseline:
step 2000: train loss 1.7648, val loss 1.8857
```
these numbers are for char-level but a similar 0.1-0.2 difference exists for token-level

## How To Run

```
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
replace model.py file with the one in this repo
python3.10 data/shakespeare_char/prepare.py
python3.10 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python3.10 sample.py --out_dir=out-shakespeare-char --device=cpu
```
