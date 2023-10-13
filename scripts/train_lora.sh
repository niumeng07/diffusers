TRAIN_IMAGE_DIR='/data1/liuda/tmp/diffusers/dataset/liudehua/'

MODEL_NAME='runwayml/stable-diffusion-v1-5'

python3.9 examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_IMAGE_DIR \
  --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-liudehua-model-lora" \
  --validation_prompt="cute dragon creature"
