import models


def build_model(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]
    #print(param)

    model = models.__dict__[cfg.type](**param)
    #print(model)
    return model
