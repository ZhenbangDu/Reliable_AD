echo "----------sampling----------"

python sample.py \
--base_model_path "digiplay/majicMIX_realistic_v7" \
--controlnet_model_path  "lllyasviel/control_v11p_sd15_canny" \
--batch_size 10 \
--sampler_name 'DDIM'  \
--num_inference_steps 40 \
--config ./config/config.json  \
--data_path  ./examples \
--save_path ./outputs
