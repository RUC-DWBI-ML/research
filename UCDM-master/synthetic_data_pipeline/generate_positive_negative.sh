#!/bin/bash
echo "Generate data for CIFAR-10"
wait

nohup python -u Dmain.py \
--distributed \
--gpu "0,1,2,3,4,5" \
--save_path "/data/cifar10/cifar10.pt.gz" \
--model_path "/data/stable-diffusion-2-base" \
--data_dir "/data" \
--data_name "cifar10" \
--known_class "0, 1" \
--unknown_class "2, 3, 4, 5, 6, 7" \
--new_class "8, 9" \
--test_size "2000, 2000, 2000" \
--class_prompt "airplane, automobile" \
--num_gpus 6 \
--batch_num 12 \
--batch_size 72 \
> generation.log 2>&1 &

echo "Runing..."