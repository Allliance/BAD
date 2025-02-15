import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
from BAD.attacks.ood.pgdlinf import PGD
from torch.utils.data import Subset
import gc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def split_dataset_by_arch(dataset):
    indices = defaultdict(lambda : [])
    all_archs = set()
    for i, model_data in enumerate(dataset.model_data):
        indices[model_data['arch']].append(i)
        all_archs.add(model_data['arch'])
    
    # Reorder the indices in each arch, so that the labels are alternating: 0, 1, 0 , 1, ...
    # To do this, first obtain the indices of the models with label 0 and 1
    # Then, reorder them by taking the first element of the first list, then the first element of the second list, then the second element of the first list, etc.
    for arch in all_archs:
        indices[arch] = sorted(indices[arch])
        label_0_indices = [i for i in indices[arch] if dataset.model_data[i]['label'] == 0]
        label_1_indices = [i for i in indices[arch] if dataset.model_data[i]['label'] == 1]
        new_indices = []
        for i in range(min(len(label_0_indices), len(label_1_indices))):
            new_indices.append(label_0_indices[i])
            new_indices.append(label_1_indices[i])
            
        if len(label_0_indices) > len(label_1_indices):
            new_indices += label_0_indices[len(label_1_indices):]
        else:
            new_indices += label_1_indices[len(label_0_indices):]
            
        indices[arch] = new_indices
    
    return {
        arch: Subset(dataset, arch_indices) for arch, arch_indices in indices.items()
    }

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

def find_min_eps(evaluator, thresh, eps_lb=0, eps_ub=1, max_error=1e-3, proportional=False, log=False):
    initial_perf = evaluator(None)
    if proportional:
        thresh *= initial_perf
        
        if log:
            print(f"Initial perf: {initial_perf}")
            print(f"Proportional threshold: {thresh}")
    
    l = eps_lb
    r = eps_ub
    
    while r-l > max_error:
        if log:
            print(f"l: {l}, r: {r}")
        mid = (r+l)/2
        auc = evaluator(mid)
        if log:
            print(f"eps: {mid}, auc: {auc}")
        if auc < thresh:
            r = mid
        else:
            l = mid
    return l

def update_attack_params(attack_dict, eps=None, steps=None):
    if eps is not None:
        attack_dict['eps'] = eps
    if steps is not None:
        attack_dict['steps'] = steps
    attack_dict['alpha'] = 2.5 * attack_dict['eps'] / attack_dict['steps']
    return attack_dict

def find_best_gap(m1, m2, evaluator, config, thresh=0.4, log=False):
    
    print("Working on config:", config['title'])
    
    best_result = {
        1: 0,
        2: 0,
        'gap': -100,
    }
    
    attack_class = config['attack']
    
    eps_lb = config.get('eps_lb')
    if eps_lb is None:
        eps_lb = 0
    
    eps_ub = config.get('eps_ub')
    if eps_ub is None:
        eps_ub = find_eps_upperbound(lambda eps: 
            evaluator(m1, attack=attack_class(m1, **(get_attack_params(eps) | config['attack_params']))), thresh, log=log)
        
    eps_steps = config.get('eps_steps')
    if eps_steps is None:
        eps_steps = 10
    
    
    epsilons = torch.linspace(eps_lb, eps_ub, eps_steps * int(255 * eps_ub)).tolist()
    gaps = []

    for eps in epsilons:
        if log:
            print("Working on epsilon", eps * 255)
        
        attack_params = get_attack_params(eps) | config['attack_params']
        
        attack1 = attack_class(m1, **attack_params)
        attack2 = attack_class(m2, **attack_params)

        score1 = evaluator(m1, attack1)
        score2 = evaluator(m2, attack2)
        
        gap = score1 - score2
        
        gaps.append(gap)
        
        if log:
            print(f'Score 1: {score1}')
            print(f'Score 2: {score2}')

        if gap > best_result['gap']:
            best_result['gap'] = gap
            best_result[1] = score1
            best_result[2] = score2
            
            print(f"{config['title']} --- Best gap until eps = {eps * 255} is {best_result['gap']}")    
    
    plot_process([e*255 for e in epsilons], gaps, config['title'])
    return best_result

def get_mean_features(model, dataloader, target_label):
    in_features = None
    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)
        data_features = model.get_features(data).detach().cpu()
        new_features = torch.index_select(data_features, 0,
                                          torch.tensor([i for i, x in enumerate(labels) if x]))
        if in_features is not None:
            in_features = torch.cat((in_features, new_features))
        else:
            in_features = new_features
    return torch.mean(in_features, dim=0)

def clear_memory():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        gc.collect()


def cosine_similaruty(A, B):
    cosine = np.dot(A, B)/(norm(A)*norm(B))
    return cosine

def get_features_mean_dict(loader, feature_extractor, progress=False):
    embeddings_dict = {}
    counts_dict = {}
    if progress:
        loader = tqdm(loader)
        
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        features = feature_extractor(data, target).detach().cpu().numpy()
        for i in range(len(target)):
            label = target[i].item()
            if label not in embeddings_dict:
                embeddings_dict[label] = features[i]
                counts_dict[label] = 1
            else:
                embeddings_dict[label] += features[i]
                counts_dict[label] += 1
    
    mean_embeddings_dict = {}
    for label in embeddings_dict:
        mean_embeddings_dict[label] = (embeddings_dict[label] / counts_dict[label])
    
    return mean_embeddings_dict

def get_ood_outputs(model, loader, DEVICE, progress=False, attack=None):
    outputs = []

    labels = []
    
    model.eval()
    model.to(device)

    
    progress_bar = loader
    if progress:
        progress_bar = tqdm(loader, unit="batch")
        
    for data, label in progress_bar:
        data, label = data.to(DEVICE), label.to(DEVICE)
        if attack:     
            data = attack(data, label)
        output = model(data)
        output = output[label==10]
        output = torch.softmax(output, dim=1)
        outputs.append(output.detach().cpu())
    o = torch.cat(outputs, dim=0)

    return o
