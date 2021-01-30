export CUDA_VISIBLE_DEVICES=0
python experiment/panet/run_PANet.py \
        --seed=666 \
        --mode=inference \
        --dataset_name=PituitaryAdenomaCls_PANet \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=PANet \
        --in_channels=5 \
        --num_classes=2 \
        --loss_name=label_smooth \
        --result=result \
        --restore=../Checkpoint/PANet/checkpoint/PANet.best.pt