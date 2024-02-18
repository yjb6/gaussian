# n3d
CUDA_VISIBLE_DEVICES=2 python test.py  --eval --skip_train \
--valloader colmap4dgsvalid --configpath configs/n3d_lite/coffee_martini_test.json \
--model_path log/flow_full_300_colororder0/ \
--source_path ../datasets/neural_3D/coffee_martini/colmap_0/ --use_loader

# hypernerf
CUDA_VISIBLE_DEVICES=2 python test.py  --eval --skip_train \
--valloader hypernerf --configpath configs/n3d_lite/coffee_martini_test.json \
--model_path log/interp-chicken/flow-full-300_color/ \
--source_path ../datasets/hy/ --use_loader