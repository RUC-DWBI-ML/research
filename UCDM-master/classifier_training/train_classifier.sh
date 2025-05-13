#!/bin/bash
echo "Classifier training..."
wait
nohup python -u Cmain.py \
--mismatch 0.2 \
--data_name "cifar10" \
--epochs 400 \
--select_epoch 40 \
--lr 5e-3 \
--batch_size 32 \
--seed 2 \
--load_path "/data/cifar10/cifar10.pt.gz" \
--data_dir /data \
--lambda1 1 \
--lambda2 2 \
--threshold 0.98 \
--config ./config/cifar10_config.json \
 > classifier.log 2>&1 &

echo "Runing..."


