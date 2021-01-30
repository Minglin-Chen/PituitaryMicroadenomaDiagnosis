from .PituitaryAdenomaDet import PituitaryAdenomaDet


def dataset_provider(name, dataset_root, is_train=True):

    if name == 'PituitaryAdenomaDet':
        dataset = PituitaryAdenomaDet(dataset_root, split='train' if is_train else 'val')
    else:
        raise ValueError
    
    return dataset