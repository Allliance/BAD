import torch
from collections import defaultdict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_best_acc_and_thresh(labels, scores):
    pairs = sorted(list(zip(scores, labels)))
    best_thresh = None
    best_acc = 0
    
    cnt = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
    for score, label in pairs:
        cnt[label][score] += 1
    
    # we keep threshold under the lowes
    correct = sum(labels)
    scores = sorted(scores)
    
    best_thresh = 0
    best_acc = correct / len(scores)
    
    for current_thresh in scores:
        correct += cnt[0][current_thresh] - cnt[1][current_thresh]
        new_acc = correct / len(scores)
        if new_acc > best_acc:
            best_acc = new_acc
            best_thresh = current_thresh
    return best_acc, best_thresh


def get_scores(model_dataset,
               score_function,
               progress):
    labels = []
    scores = []

    tq = range(len(model_dataset))
    tq = tqdm(tq)
    
    for i in tq:
        model, _, _ = model_dataset[i]
        
        name = model.meta_data.get('attack')
        
        if name is None:
            label = 1
        else:
            label = 0

        score = score_function(model)

        if progress:
            print(label, score)
        
        scores.append(score)
        labels.append(label)
    
    return scores, labels


def find_best_eps(eps_lb, eps_ub, eps_step, validation_function, max_error=1e-3, partition=10):
    '''
    [eps_lb, eps_ub]: determines the search space for epsilon
    eps_step: step size for epsilon, for example when eps_lb = 0, eps_ub = 0.4, eps_step = 0.1, the search space is [0, 0.1, 0.2, 0.3]
    validation_function: a function that takes hyperparameters (here only epsilon) as input and returns
    a score(it can be auc on auc, auc on l2, auc on kld, etc.).
    The given function for example iterates over a validation model set,
    computes some scores for each model using the given hyperparameters (here epsilon only), and then returns auc of the model set on that score.
    max_error: the maximum error allowed for the best epsilon, if the step size is greater than max_error,
    the function will recursively call itself with a smaller step size.
    partition: this number is used when doing the recursive call. If the step size is larger than max_error,
    then should work on more fine-grained epsilons, hence we have to partition the step size into smaller steps.
    When doing the recursive call, the new step size will be eps_step/partition.
    '''
    
    current_eps = eps_lb
    all_scores = []
    while current_eps < eps_ub:
        print(f"Testing epsilon: {current_eps * 255}/255")
        new_score = validation_function(current_eps, progress=False)
        all_scores.append((new_score, current_eps))
        current_eps += eps_step
    
    all_scores = sorted(all_scores)
    best_eps = all_scores[-1][-1]
    
    if eps_step > max_error:
        max_error /= partition
        new_eps_step = eps_step/partition
        return find_best_eps(eps_lb=max(best_eps - eps_step + new_eps_step, 0),
                             eps_ub=best_eps + eps_step - new_eps_step,
                             eps_step=new_eps_step,
                             validation_function=validation_function,
                             max_error=max_error,
                             partition=partition)
    
    return best_eps
    
best_eps = find_best_eps(0, 0.4/255, 0.1/255, get_auc_on_auc)
