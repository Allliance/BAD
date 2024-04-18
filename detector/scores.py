import torch
import torchvision
import numpy as np

from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate
from BAD.utils import update_attack_params, get_features

def epsilon_score(attack_class, attack_params, evaluator, eps_config, log=False):
    initial_perf = evaluator(None)
    eps_lb = eps_config['eps_lb']
    eps_ub = eps_config['eps_ub']
    eps_step = eps_config['eps_step']
    perf_thresh = eps_config['perf_thresh'] * initial_perf
    print("Initial perf:", initial_perf)
    current_eps = eps_lb
    current_step = current_eps
    back = False
    while (not back) or current_step > eps_step:
        attack_params = update_attack_params(attack_params, current_eps)
        attack = attack_class(**attack_params)
        perf = evaluator(attack)
        if perf < perf_thresh:
            back = True
            current_step /= 2
            current_eps -= current_step
        else:
            current_step *= 2
            current_eps += current_step
    
    return current_eps


def get_adv_features(model, loader, target, mean_embeddings, attack, progress=False, ):
    features = []
    labels = []
    
    model.eval()
    model.to(device)

    
    progress_bar = loader
    if progress:
        progress_bar = tqdm(loader, unit="batch")
        
    for data, label in progress_bar:
        labels += label.tolist()
        data, label = data.to(device), label.to(device)
        if attack is not None:
            data = attack(data, label)
        feature = model.get_features(data)
        c_f = feature.detach().cpu().numpy()
        features.append(c_f)
    features = np.concatenate(features)
    
    labels = np.array(labels)
    out_features = features[1 - labels]
    in_features = features[labels]

    return out_features, in_features

# score in [l2, cosine]
def max_diff(model, testloader, attack_class=None, attack_params=None, score='l2', use_in=True, progress=False):
    max_l2 = 0
    
    v_out_b, v_in_b = get_features(model, testloader, attack=None)
    v_in_b_mean = np.mean(v_in_b, axis=0)
    v_out_b_mean = np.mean(v_out_b, axis=0)
    if attack_params.get('target_class') is not None:
        best_target = None
        tq = range(10)
        if progress:
            tq = tqdm(range(10))
        for i in tq:
            attack_params['target_class'] = i
            attack = attack_class(**attack_params)
            v_out_a, v_in_a = get_features(model, testloader, attack=attack)
            v_out_a_mean = np.mean(v_out_a, axis=0)
            v_in_a_mean = np.mean(v_in_a, axis=0)
            if use_in:
                diff_a = (v_out_a_mean - v_in_a_mean)
                diff_b = (v_out_b_mean - v_in_b_mean)
                #cosine = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
                l2 = norm(diff_a - diff_b)     
                if l2 > max_l2:
                    max_l2 = l2
            else:
                diff = v_out_a_mean - v_out_b_mean
                l2 = norm(diff)
                if l2 > max_l2:
                    max_l2 = l2
                    best_target = i
        return best_target, max_l2
    else:
        attack = attack_class(**attack_params)
        v_out_a, v_in_a = get_features(model, testloader,  attack=attack)
        v_out_a_mean = np.mean(v_out_a, axis=0)
        v_in_a_mean = np.mean(v_in_a, axis=0)
        if use_in:
            diff_a = (v_out_a_mean - v_in_a_mean)
            diff_b = (v_out_b_mean - v_in_b_mean)
            #score = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
            score = norm(diff_a - diff_b)
        else:
            diff = v_out_a_mean - v_out_b_mean
            score = norm(diff)
        return score
    

