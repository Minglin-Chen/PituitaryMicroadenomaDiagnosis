from .PituitaryAdenomaCls import PituitaryAdenomaCls
from .PituitaryAdenomaCls3D import PituitaryAdenomaCls3D
from .PituitaryAdenomaCls_PANet import PituitaryAdenomaCls_PANet


def dataset_provider(name, dataset_root, is_train=True):

    if name == 'PituitaryAdenomaCls':
        dataset = PituitaryAdenomaCls(dataset_root, split='train' if is_train else 'val')
    elif name == 'PituitaryAdenomaCls3D':
        dataset = PituitaryAdenomaCls3D(dataset_root, split='train' if is_train else 'val')
    elif name == 'PituitaryAdenomaCls_PANet':
        dataset = PituitaryAdenomaCls_PANet(dataset_root, split='train' if is_train else 'val')
    else:
        raise ValueError

    return dataset