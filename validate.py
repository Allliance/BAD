import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# this is a general function for getting scores from a model dataset
# Don't care about it
def get_models_scores(model_dataset,
               model_score_function,
               progress):
    '''
    This is a general function for getting scores from a model dataset.
    model_dataset: an iterable that contains models
    model_score_function: a function that takes a model as input and returns a score
    progress: if True, it will print the progress of the function
    '''
    labels = []
    scores = []

    tq = range(len(model_dataset))
    tq = tqdm(tq)
    
    for i in tq:
        model, label = model_dataset[i]

        score = model_score_function(model)

        if progress:
            print(label, score)
        
        scores.append(score)
        labels.append(label)
    
    return scores, labels

# All it does is that it iterates over the model dataset, calculate some score bease on the model and the dataloader
# and finally returns auc on the scores
def get_auc_on_models_scores(model_dataset,
                           score_function,
                           dataloader,
                           other_score_function_params={},
                           dataloader_func=None,
                           progress=False):
    '''
    This function calculates the AUC of the model on the given model dataset.
    model_dataset: an iterable that contains models
    score_function: a function that takes a model as input and returns a score
    progress: if True, it will print the progress of the function
    '''
    assert dataloader is not None or dataloader_func is not None

    # this is a function that just calls the score function on the model
    # the purpose of this function is to be compatible with get_models_scores
    
    def model_score_function(model):
        final_dataloader = dataloader_func(model) if dataloader_func is not None else dataloader
        
        return score_function(model, final_dataloader, progress=progress, **other_score_function_params)
    
    scores, labels = get_models_scores(model_dataset, model_score_function, progress)
    
    return roc_auc_score(labels, scores)


def find_best_eps(eps_lb, eps_ub, eps_step, validation_function, max_error=1e-3, partition=10, progress=False, verbose=False):
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
        new_score = validation_function(current_eps, progress=False)
        if progress:
            print(f"Testing on eps={current_eps * 255}/255 finished{'' if not verbose else f' with score {new_score}'}")
        all_scores.append((new_score, current_eps))
        current_eps += eps_step
    
    all_scores = sorted(all_scores)
    best_eps = all_scores[-1][-1]
    
    if eps_step > max_error:
        max_error /= partition
        new_eps_step = eps_step/partition
        if progress:
            print(f"Best epsilon is {best_eps * 255}/255 with a score of {all_scores[-1][0]}")
            print(f"Recursively calling the function with new step size {new_eps_step*255}/255")
        return find_best_eps(eps_lb=max(best_eps - eps_step + new_eps_step, 0),
                             eps_ub=best_eps + eps_step - new_eps_step,
                             eps_step=new_eps_step,
                             validation_function=validation_function,
                             max_error=max_error,
                             partition=partition,
                             verbose=verbose)
    
    return best_eps


# Example of usage
# Assume all we have is validation model set --> val_modelset
# Assume score function is get_l2
# Assume we have testloader
# def validation_function(eps):
#     attack_eps = eps
#     attack_steps = 10
#     attack_alpha = 2.5 * attack_eps / attack_steps
    
#     def score_function(model, dataloader, progress, **kwargs):
#         attack = Attack(model, eps=attack_eps, steps=attack_steps, alpha=attack_alpha)
#         return get_l2(model, dataloader, attack=attack, progress=progress, **kwargs)
    
#     return get_auc_on_models_scores(val_modelset, score_function, testloader, progress=False)

# best_eps = find_best_eps(0, 8/255, 1/255, validation_function, max_error=1e-3, partition=10)
# print("Best epsilon is:", best_eps * 255)
