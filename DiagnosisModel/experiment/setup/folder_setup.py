import os
import time


def folder_setup(args):

    time_id = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_root, args.dataset_name+'_'+args.model, time_id)
    assert not os.path.exists(log_dir)
    os.makedirs(log_dir)

    # checkpoint directory
    ckpt_dir = os.path.join(log_dir, 'checkpoint')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

    # tensorboard directory
    tsrboard_dir = os.path.join(log_dir, 'tensorboard')

    return ckpt_dir, tsrboard_dir