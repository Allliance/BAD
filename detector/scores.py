import torch
import torchvision
import numpy as np

from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate
from BAD.utils import update_attack_params, get_features_mean_dict

def get_epsilon_score(attack_class, attack_params, evaluator, eps_config, log=False):
    initial_perf = evaluator(None)
    eps_lb = eps_config['eps_lb']
    eps_ub = eps_config['eps_ub']
    eps_step = eps_config['eps_step']
    perf_thresh = eps_config['perf_thresh'] * initial_perf
    print("Initial perf:", initial_perf)
    current_eps = eps_lb
    l = eps_lb
    r = eps_ub
    while r-l > eps_step:
        mid = (r+l)/2
        attack_params = update_attack_params(attack_params, mid)
        attack = attack_class(**attack_params)
        perf = evaluator(attack)
        if perf < perf_thresh:
            r = mid
        else:
            l = mid
    return l


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
def max_diff(model, testloader, attack_class=None, attack_params=None,
             score='l2', use_in=True, progress=False, num_classes=10, normalize_features=False):
    max_l2 = 0
    
    initial_features = get_features_mean_dict(testloader, feature_extractor=lambda data, targets: model.get_features(data, normalize_features))
    in_features = initial_features[1]
    out_features = initial_features[0]
    
    mean_in_initial_features = np.mean(in_features, axis=0)
    mean_out_initial_features = np.mean(out_features, axis=0)
    initial_diff = (mean_out_initial_features - mean_in_initial_features)
    
    def get_adv_feature_extractor(attack):
        return lambda data, targets : model.get_features(attack(data, targets), normalize_features)
    
    if attack_params.get('target_class') is not None:
        best_target = None
        tq = range(10)
        if progress:
            tq = tqdm(range(10))
        for i in tq:
            attack_params['target_class'] = i
            attack = attack_class(**attack_params)
            adv_features = get_features_mean_dict(testloader, get_adv_feature_extractor(attack))
            in_adv_features = adv_features[1]
            out_adv_features = adv_features[0]
            mean_in_adv_features = np.mean(in_adv_features, axis=0)
            mean_out_adv_features = np.mean(out_adv_features, axis=0)
            if use_in:
                adv_diff = (mean_out_adv_features - mean_in_adv_features)
                #cosine = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
                l2 = norm(adv_diff - initial_diff)     
                if l2 > max_l2:
                    max_l2 = l2
            else:
                diff = mean_out_adv_features - mean_out_initial_features
                l2 = norm(diff)
                if l2 > max_l2:
                    max_l2 = l2
                    best_target = i
        return best_target, max_l2
    else:
        attack = attack_class(**attack_params)
        adv_features = get_features_mean_dict(testloader, get_adv_feature_extractor(attack))
        in_adv_features = adv_features[1]
        out_adv_features = adv_features[0]
        mean_in_adv_features = np.mean(in_adv_features, axis=0)
        mean_out_adv_features = np.mean(out_adv_features, axis=0)
        if use_in:
            adv_diff = (mean_out_adv_features - mean_in_adv_features)
            #score = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
            score = norm(adv_diff - initial_diff)
        else:
            diff = mean_out_adv_features - mean_out_initial_features
            score = norm(diff)
        return score
    

