import transformers
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime
from typing import List


def get_time_str():
    currentDateAndTime = datetime.now()
    day = currentDateAndTime.strftime("%D").replace('/', '-')
    time = currentDateAndTime.strftime("%H:%M:%S")
    currentTime = day + '/' + time
    return currentTime

def get_callbacks(ckp_config):
    checkpoint_callback = ModelCheckpoint(**ckp_config)
    callbacks = []
    callbacks.append(LearningRateMonitor('step'))
    callbacks.append(checkpoint_callback)
    return callbacks


def get_step_lr_sche(optimizer, 
                     epoches: List[int],
                     muls: List[float]):
    """at epoches, mul lr with muls"""
    def step_lr_fn(epoch, epoches: List[int], muls: List[float]):
        if len(epoches) == 1:
            if epoch < epoches[0]:
                return 1.0
            else:
                return muls[0]
        elif len(epoches) == 2:
            if epoch < epoches[0]:
                return 1.0
            elif epoch < epoches[1]:
                return muls[0]
            else:
                return muls[0] * muls[1]
        else:
            raise NotImplementedError
    from functools import partial
    fn = partial(step_lr_fn, epoches=epoches, muls=muls)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    return scheduler    

def get_opt_lr_sch(optimizer_config, lr_sche_config, model):
    if optimizer_config.type == 'sgd':
        optimizer = optim.SGD
    elif optimizer_config.type == 'adam':
        optimizer = optim.Adam
    elif optimizer_config.type == 'adamw':
        optimizer = optim.AdamW
    else:
        raise NotImplementedError
    
    optimizer = optimizer(model.parameters(),
                            **optimizer_config.config)
    if lr_sche_config.type == 'constant':
        lr_sche = transformers.get_constant_schedule(optimizer)
    elif lr_sche_config.type == 'linear':
        lr_sche = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                **lr_sche_config.config)
    elif lr_sche_config.type == 'cosine':
        lr_sche = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                **lr_sche_config.config)
    elif lr_sche_config.type == 'step':
        lr_sche = get_step_lr_sche(optimizer=optimizer,
                                   **lr_sche_config.config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_sche,
                'interval': 'epoch'
            }
        }
    elif lr_sche_config.type == 'reduce':
        lr_sche = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        **lr_sche_config.config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_sche,
                'monitor': lr_sche_config.config.reduce_monitor,
                'interval': 'step'
            }
        }
    else:
        raise NotImplementedError
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': lr_sche,
            'interval': 'step'
        }
    }
     
