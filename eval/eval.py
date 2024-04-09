from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from ..scores.msp import get_msp

def eval_cls(model, loader, device, attack=None, progress=False):
    
    model.to(device)
    model.eval()
    correct = 0
    
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

    test_acc = correct / len(loader.dataset)
    return test_acc


def eval_ood(model, loader, target=None, progress=False, attack_eps=0):
    all_scores = []
    all_labels = []
    
    attack_steps = 10
    attack_alpha = 2.5 * attack_eps / attack_steps
    
    model.eval()
    model.to(device)

    attack = None
    if attack_eps>0:
        attack = Attack(model, target_class=target, eps=attack_eps, alpha=attack_alpha, steps=attack_steps)
    
    progress_bar = loader
    if progress:
        progress_bar = tqdm(loader, unit="batch")
        
    for data, label in progress_bar:
        data, label = data.to(device), label.to(device)
        if attack is not None:
            data = attack(data, label)
        scores = get_msp(model, data)
        all_scores += scores.tolist()
        all_labels += label.tolist()

    auroc = roc_auc_score(all_labels, all_scores)
    return auroc