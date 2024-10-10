torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "logs/lora_openvla-7b_libero_spatial_imageinimage/spatial-5000steps" \
  --data_root_dir datasets/libero-datasets \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir logs/lora_openvla-7b_libero_spatial_imageinimage \
  --adapter_tmp_dir logs/ft_openvla-7b_libero_spatial_imageinimage/lora_tmp \
  --lora_rank 32 \
  --batch_size 6 \
  --grad_accumulation_steps 10 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project openvla-7b-lora-picinpic \
  --save_steps 5000 \
  --max_steps 40000

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "logs/lora_openvla-7b_libero_spatial_imageinimage/spatial-5000steps" \
  --data_root_dir datasets/libero-datasets \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir logs/lora_openvla-7b_libero_spatial_imageinimage \
  --adapter_tmp_dir logs/ft_openvla-7b_libero_spatial_imageinimage/lora_tmp \
  --lora_rank 32 \
  --batch_size 6 \
  --grad_accumulation_steps 10 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project openvla-7b-lora-picinpic \
  --save_steps 5000 \
  --max_steps 40000