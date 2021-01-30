export CUDA_VISIBLE_DEVICES=0
python experiment/fasterrcnn/run_FasterRCNN.py \
        --seed=666 \
        --mode=train \
        --log_root=logs \
        --dataset_name=PituitaryAdenomaDet \
        --dataset_root=../Dataset/Detection/trainval_index.npz \
        --batch_size=16 \
        --model=FasterRCNN \
        --num_classes=2 \
        --lr=5e-3 \
        --weight_decay=5e-4 \
        --num_epoch=20