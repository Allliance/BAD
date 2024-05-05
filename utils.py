import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from numpy.linalg import norm
from tqdm import tqdm
from eval.eval import evaluate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
from attacks.ood.pgdlinf import PGD


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
