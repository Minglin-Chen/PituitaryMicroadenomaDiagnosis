export CUDA_VISIBLE_DEVICES=0
python experiment/cnn3d/run_CNN3D.py \
        --seed=666 \
        --mode=evaluate \
        --dataset_name=PituitaryAdenomaCls3D \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=CNN3D \
        --num_classes=2 \
        --pretrained='' \
        --loss_name=ce \
        --restore=../Checkpoint/CNN3D/CNN3D/checkpoint/CNN3D.best.pt