import torch
import torchvision
import numpy as np

from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate

def get_features(model, loader, attack=None, progress=False):
    features = []
    outputs = []

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
            data = attack(data, torch.where(label == 10, torch.tensor(0), torch.tensor(1)))
        feature = model.get_features(data)
        output = model(data)
        output = torch.softmax(output, dim=1)
        c_f = feature.detach().cpu().numpy()
        c_o = output.detach().cpu().numpy()
        features.append(c_f)
        outputs.append(c_o)
    f = np.concatenate(features)
    o = np.concatenate(outputs)
    mask_o = np.array(labels)== 10
    mask_i = np.array(labels)!= 10
    gaussian_features = f[mask_o]
    cifar_features = f[mask_i]
    selected_out = o[mask_o]

    return gaussian_features, cifar_features

# score in [l2, cosine]
def get_max_diff(model, testloader, attack_config=None, score='l2', use_in=True, progress=False):
    max_l2 = 0
    
    attack_class = attack_config['attack_class']
    attack_config.pop('attack_class')
    
    v_out_b, v_in_b = get_features(model, testloader, attack=None)
    v_in_b_mean = np.mean(v_in_b, axis=0)
    v_out_b_mean = np.mean(v_out_b, axis=0)
    if attack_config.get('target_class') is not None:
        best_target = None
        tq = range(10)
        if progress:
            tq = tqdm(range(10))
        for i in tq:
            attack_config['target_class'] = i
            attack = attack_class(model, **attack_config)
            v_out_a, v_in_a = get_features(model, testloader,  attack=attack)
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
        attack = attack_class(model, **attack_config)
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
    

