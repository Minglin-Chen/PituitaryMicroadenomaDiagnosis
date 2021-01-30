from .FasterRCNN import fasterrcnn


model_park = {
    'FasterRCNN': fasterrcnn}


def model_provider(name, **kwargs):

    model_ret = model_park[name](**kwargs)
    
    return model_ret