export CUDA_VISIBLE_DEVICES=0
python experiment/baseline/run_CNN.py \
        --seed=666 \
        --mode=train \
        --log_root=logs \
        --dataset_name=PituitaryAdenomaCls \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=baseline \
        --model_name=resnet50 \
        --in_channels=5 \
        --num_classes=2 \
        --loss_name=ce \
        --lr=1e-2 \
        --weight_decay=5e-4 \
        --num_epoch=500 \
        --pretrained