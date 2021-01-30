export CUDA_VISIBLE_DEVICES=0
python experiment/panet/run_PANet.py \
        --seed=666 \
        --mode=train \
        --log_root=logs \
        --dataset_name=PituitaryAdenomaCls_PANet \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=PANet \
        --in_channels=5 \
        --num_classes=2 \
        --loss_name=label_smooth \
        --lr=1e-2 \
        --weight_decay=5e-4 \
        --num_epoch=500 \
        --pretrained