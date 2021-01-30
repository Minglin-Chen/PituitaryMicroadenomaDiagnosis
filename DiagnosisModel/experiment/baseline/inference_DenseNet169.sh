export CUDA_VISIBLE_DEVICES=0
python experiment/baseline/run_CNN.py \
        --seed=666 \
        --mode=inference \
        --dataset_name=PituitaryAdenomaCls \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=baseline \
        --model_name=densenet169 \
        --in_channels=5 \
        --num_classes=2 \
        --loss_name=ce \
        --restore=../Checkpoint/baseline/densenet169/checkpoint/densenet169.best.pt \
        --result=result