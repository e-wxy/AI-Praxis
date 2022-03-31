import torch
import os


# store model

def load_model(device, path='.', name='model.pkl'):
    """
    load model from path/model/name 加载网络
    """
    pth_model = os.path.join(path, 'model', name)
    assert os.path.exists(pth_model), "Model file doesn't exist!"
    model = torch.load(pth_model, map_location=device)
    print('Load {} on {} successfully.'.format(name, device))
    return model
    

def save_model(model, path='.', name='model.pkl'):
    """ 
    save model to path/model/name 保存网络
    """

    model_dir = os.path.join(path, 'model')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      
    pth_model = os.path.join(model_dir, name)
    torch.save(model, pth_model)
    print('Model has been saved to {}'.format(pth_model))


def save_state_dict(model, path='.', name='state_dict.pth'):
    """ 
    save state dict to path/model/temp/name 保存网络参数
    """

    model_dir = os.path.join(path, 'model', 'temp')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      
    pth_dict = os.path.join(model_dir, name)
    torch.save(model.state_dict(), pth_dict)
    print('State dict has been saved to {}'.format(pth_dict))
    
    
def load_state_dict(model, device, path='.', name='state_dict.pth'):
    """ 
    load model parmas from state_dict 加载网络参数
    """
    pth_dict = os.path.join(path, 'model', 'temp', name)
    assert os.path.exists(pth_dict), "State dict file doesn't exist!"
    model.load_state_dict(torch.load(pth_dict, map_location=device))
    return model


# checkpoint

def check_train(log, model, optimizer, epoch, scheduler=None, pth_check='ch_training.pth'):
    """ save training checkpoint
        保存训练参数：model, epoch, optimizer, schedluer

    Args:
        log (Logger)
        pth_check (str): path to store the checkpoint.
    """
    check_dir = 'checkpoint'
    if not os.path.exists(check_dir):
      os.makedirs(check_dir)
    pth_check = os.path.join(check_dir, pth_check)

    log.logger.info("Saving training checkpoint at {}".format(pth_check))
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, pth_check)


def check_eval(log, costs, train_accs, test_accs, b_accs, f1_scores, pth_check='ch_eval.pth', verbose=True):
    """ saving evaluation checkpoint
        保存训练过程的cost, accs, f1-score

    Args:
        log (Logger)
        pth_eval (str): path to store the checkpoint.
        verbose: whether showing details
    """
    check_dir = 'checkpoint'
    if not os.path.exists(check_dir):
      os.makedirs(check_dir)
    pth_check = os.path.join(check_dir, pth_check)
    
    if verbose:
        log.logger.info("Saving training checkpoint at {}".format(pth_check))
    checkpoint = {
        'costs': costs,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'b_accs': b_accs,
        'f1_scores': f1_scores,
    }

    if verbose:
        for key in checkpoint.keys(): 
            log.logger.info('{} = {}\n'.format(key, checkpoint[key]))

    torch.save(checkpoint, pth_check)


def load_train(log, model, optimizer, scheduler=None, pth_check=None):
    """ initialize or load training process from checkpoint
        从checkpoint加载训练状态，pth_check为None时，进行初始化

    Args:
        log (Logger)
        pth_check (str): path of training checkpoint file. e.g. 'ch_training.pth'. (Default: None - 初始化)

    Returns:
        start epoch
    """
    if pth_check == None:
        return 0
    
    pth_check = os.path.join('checkpoint', pth_check)
    log.logger.info("Reloading training checkpoint from {}".format(pth_check))
    checkpoint = torch.load(pth_check)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return start_epoch

def load_eval(log, pth_check=None):
    """ initialize or load evaluation from checkpoint
        从checkpoint加载之前训练过程的模型表现，pth_check为None时，进行初始化

    Args:
        log (Logger)
        pth_check (str): path of eval checkpoint file. e.g. 'ch_eval.pth'
        robust (bool): whether having robustness metrics.

    Returns:
        costs, train_accs, test_accs, b_accs, f1_scores, (robust_accs, robust_baccs, robust_f1s)
    """

    if pth_check == None:
        return [], [], [], [], []

    pth_check = os.path.join('checkpoint', pth_check)
    log.logger.info("Reloading evaluation checkpoint from {}".format(pth_check))
    checkpoint = torch.load(pth_check)

    costs = checkpoint['costs']
    train_accs = checkpoint['train_accs']
    test_accs = checkpoint['test_accs']
    b_accs = checkpoint['b_accs']
    f1_scores = checkpoint['f1_scores']
    
    return costs, train_accs, test_accs, b_accs, f1_scores
