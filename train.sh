
# hypernerf
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
--eval --config configs/hyper/chicken.json --model_path log/interp-chicken/flow-full-color4/ \
--source_path ../datasets/hypernerf/chickchicken/  --static_iteration 2000 --color_order 4\
>> log/interp-chicken/flow-full-color4/nohup.out 2>&1 &

#n3d
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
--eval --config configs/n3d_lite/coffee_martini_test.json --model_path log/flame_steak/flow_full_color0 \
--source_path ../datasets/neural_3D/flame_steak/colmap_0/  --static_iteration 2000 --use_loader \
 >>log/flame_steak/flow_full_color0/nohup.out 2>&1 &