from BAD.eval.eval import evaluate
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_auc(model, dataloader, attack=None, progress=False):
    '''
    This function calculates the AUC of the model on the given dataloader.
    model: the model to be evaluated
    dataloader: the dataloader that contains the data to be evaluated
    attack: an attack function that takes the data and the label as input and returns the adversarial example
    '''
    
    return evaluate(model=model, loader=dataloader, device=device, attack=attack, metric='auc', progress=progress)

