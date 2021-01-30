from .baseline.CNN import CNN
from .cnn3d.CNN3D import CNN3D
from .panet.PANet import PANet


model_park = {
    'baseline': CNN,
    'CNN3D': CNN3D,
    'PANet': PANet}


def model_provider(name, **kwargs):

    model_ret = model_park[name](**kwargs)
    
    return model_ret