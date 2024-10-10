torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "hf/mimicgen" \
  --data_root_dir /data2/peiyh/tensorflow_datasets \
  --dataset_name prepare_coffee_orig \
  --run_root_dir logs/lora_openvla-7b_prepare_coffee \
  --adapter_tmp_dir logs/lora_openvla-7b_prepare_coffee/lora_tmp \
  --lora_rank 32 \
  --batch_size 6 \
  --grad_accumulation_steps 10 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project openvla-7b-lora-prepare_coffee \
  --save_steps 10000 \
  --max_steps 10000

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "hf/openvla" \
  --data_root_dir /data2/peiyh/tensorflow_datasets \
  --dataset_name prepare_coffee_orig \
  --run_root_dir logs/lora_openvla-7b_prepare_coffee \
  --adapter_tmp_dir logs/lora_openvla-7b_prepare_coffee/lora_tmp \
  --lora_rank 32 \
  --batch_size 6 \
  --grad_accumulation_steps 10 \
  --learning_rate 2e-4 \
  --image_aug True \
  --wandb_project openvla-7b-lora-prepare_coffee \
  --save_steps 10000 \
  --max_steps 10000