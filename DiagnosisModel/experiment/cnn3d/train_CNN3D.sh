export CUDA_VISIBLE_DEVICES=0
python experiment/cnn3d/run_CNN3D.py \
        --seed=666 \
        --mode=train \
        --log_root=logs \
        --dataset_name=PituitaryAdenomaCls3D \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=CNN3D \
        --num_classes=2 \
        --loss_name=ce \
        --lr=1e-2 \
        --weight_decay=5e-4 \
        --num_epoch=500 \
        --pretrained='../Checkpoint/Genesis_Chest_CT.pt'