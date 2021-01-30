export CUDA_VISIBLE_DEVICES=0
python experiment/baseline/run_CNN.py \
        --seed=666 \
        --mode=inference \
        --dataset_name=PituitaryAdenomaCls \
        --dataset_root=../Dataset/Diagnosis/trainval_index.npz \
        --batch_size=16 \
        --model=baseline \
        --model_name=resnext50_32x4d \
        --in_channels=5 \
        --num_classes=2 \
        --loss_name=ce \
        --restore=../Checkpoint/baseline/resnext50_32x4d/checkpoint/resnext50_32x4d.best.pt \
        --result=result