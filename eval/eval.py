from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from ..scores.msp import get_msp

def eval(model, loader, device, metric='acc', attack=None, progress=False):
    model.to(device)
    model.eval()
    correct = 0
    all_scores = []
    all_labels = []
    
    progress_bar = loader
    if progress:
        progress_bar = tqdm(loader, unit="batch")
    
    for data, target in progress_bar:
        data = data.to(device)
        target = target.to(device)

        if attack:
            data = attack(data, target)

        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        scores = get_msp(model, data)
        all_scores += scores.tolist()
        all_labels += target.tolist()


    if metric == 'acc':
        return correct / len(loader.dataset)
    elif metric == 'auc':
        return roc_auc_score(all_labels, all_scores)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")