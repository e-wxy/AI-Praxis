import torch
import os


# store model

def load_model(device, path='.', name='model.pkl'):
    pth_model = os.path.join(path, 'model', name)
    assert os.path.exists(pth_model), "Model file doesn't exist!"
    model = torch.load(pth_model, map_location=device)
    print('Load {} on {} successfully.'.format(name, device))
    return model
    

def save_model(model, path='.', name='model.pkl'):
    """ save model to path/model/name """

    model_dir = os.path.join(path, 'model')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      
    pth_model = os.path.join(model_dir, name)
    torch.save(model, pth_model)
    print('Model has been saved to {}'.format(pth_model))


def save_state_dict(model, path='.', name='state_dict.pth'):
    """ save state dict to path/model/temp/name """

    model_dir = os.path.join(path, 'model', 'temp')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      
    pth_dict = os.path.join(model_dir, name)
    torch.save(model, pth_dict)
    print('State dict has been saved to {}'.format(pth_dict))


# checkpoint

def check_train(log, model, optimizer, epoch, scheduler=None, pth_check='ch_training.pth'):
    """ save training checkpoint

    Args:
        log (Logger)
        pth_check (str): path to store the checkpoint.
    """
    log.logger.info("Saving training checkpoint at {}".format(pth_check))
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, pth_check)


def check_eval(log, costs, train_accs, test_accs, b_accs, f1_scores, pth_check='ch_eval.pth'):
    """ saving evaluation checkpoint

    Args:
        log (Logger)
        pth_eval (str): path to store the checkpoint.
    """
    log.logger.info("Saving training checkpoint at {}".format(pth_check))
    checkpoint = {
        'costs': costs,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'b_accs': b_accs,
        'f1_scores': f1_scores,
    }


    for key in checkpoint.keys(): 
        log.logger.info('{} = {}\n'.format(key, checkpoint[key]))

    torch.save(checkpoint, pth_check)


def load_train(log, model, optimizer, scheduler=None, pth_check=None):
    """ initialize or load training process from checkpoint

    Args:
        log (Logger)
        pth_check (str): path of training checkpoint file. e.g. 'ch_training.pth'

    Returns:
        start epoch
    """
    if pth_check == None:
        return 0
    
    log.logger.info("Reloading training checkpoint from {}".format(pth_check))
    checkpoint = torch.load(pth_check)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return start_epoch

def load_eval(log, pth_check=None):
    """ initialize or load evaluation from checkpoint

    Args:
        log (Logger)
        pth_check (str): path of eval checkpoint file. e.g. 'ch_eval.pth'
        robust (bool): whether having robustness metrics.

    Returns:
        costs, train_accs, test_accs, b_accs, f1_scores, (robust_accs, robust_baccs, robust_f1s)
    """

    if pth_check == None:
        return [], [], [], [], []

    log.logger.info("Reloading evaluation checkpoint from {}".format(pth_check))
    checkpoint = torch.load(pth_check)

    costs = checkpoint['costs']
    train_accs = checkpoint['train_accs']
    test_accs = checkpoint['test_accs']
    b_accs = checkpoint['b_accs']
    f1_scores = checkpoint['f1_scores']
    
    return costs, train_accs, test_accs, b_accs, f1_scores
