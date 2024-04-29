import torch
import torchvision
import numpy as np

import torch.nn.functional as F

from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate
from BAD.utils import update_attack_params, get_features_mean_dict, find_min_eps
from BAD.utils import get_ood_outputs
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def get_epsilon_score(eps_evaluator, eps_config, log=False, proportional=False):
    return find_min_eps(eps_evaluator, eps_config['thresh'], eps_lb=eps_config['lb'], 
                        eps_ub=eps_config['ub'], max_error=eps_config['max_error'], proportional=proportional, log=log)


def get_features(model, loader, attack, progress=False, ):
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
        c_f = feature.squeeze().detach().cpu().numpy()
        features.append(c_f)
    features = np.concatenate(features)
    
    labels = np.array(labels)
    

    return features, labels

# score in [l2, cosine]
def max_diff(model, testloader, attack_class=None, attack_params=None,
             score='l2', use_in=True, progress=False, num_classes=10, normalize_features=False):
    max_l2 = 0
    
    mean_initial_features = get_features_mean_dict(testloader, feature_extractor=lambda data, targets: model.get_features(data, normalize_features))
    mean_in_initial_features = mean_initial_features[1]
    mean_out_initial_features = mean_initial_features[0]

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
            mean_adv_features = get_features_mean_dict(testloader, get_adv_feature_extractor(attack))
            mean_in_adv_features = mean_adv_features[1]
            mean_out_adv_features = mean_adv_features[0]
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
        mean_adv_features = get_features_mean_dict(testloader, get_adv_feature_extractor(attack))
        mean_in_adv_features = mean_adv_features[1]
        mean_out_adv_features = mean_adv_features[0]
        if use_in:
            adv_diff = (mean_out_adv_features - mean_in_adv_features)
            #score = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
            score1 = norm(adv_diff - initial_diff)
            score2 = cosine_similaruty(adv_diff,initial_diff)
            return score1, score2
        else:
            diff = mean_out_adv_features - mean_out_initial_features
            score = norm(diff)
        return score


def cosine_similaruty(A, B):
    cosine = np.dot(A, B)/(norm(A)*norm(B))
    return cosine

def compute_kl_divergence(pdf_p, pdf_q):
    # Compute KL divergence between two distributions.
    # Add a small number to probability distributions to avoid log(0)
    epsilon = 1e-10
    pdf_p = pdf_p + epsilon
    pdf_q = pdf_q + epsilon
    return entropy(pdf_p, pdf_q)


def get_fid(features_adv, features_clean):
    mean1 = np.mean(features_adv, axis=0)
    cov1 = np.cov(features_adv, rowvar=False)

    mean2 = np.mean(features_clean, axis=0)
    cov2 = np.cov(features_clean, rowvar=False)

    mean_diff = mean1 - mean2
    mean_diff_squared = np.dot(mean_diff, mean_diff)

    cov_product = np.dot(cov1, cov2)
    cov_sqrt = linalg.sqrtm(cov_product)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = mean_diff_squared + np.trace(cov1 + cov2 - 2 * cov_sqrt)
    return fid


def KLD_TSNE_probs(all_embeddings, labels):


    # Apply t-SNE to the PCA-reduced combined embeddings
    tsne = TSNE(n_components=2, perplexity=30, n_iter=600)
    all_embeddings_tsne = tsne.fit_transform(all_embeddings)

    out_embeddings_tsne = all_embeddings_tsne[labels==0]
    in_embeddings_tsne = all_embeddings_tsne[labels==1]
    

    kde_original = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(in_embeddings_tsne)

    # Compute the density of the original embedding space
    pdf_original = np.exp(kde_original.score_samples(in_embeddings_tsne))




    pdf_out = np.exp(kde_original.score_samples(out_embeddings_tsne))

        # Compute KL divergence
    kl_div = compute_kl_divergence(pdf_original, pdf_out)

    return kl_div


def KLD_score_points(all_embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=600)
    all_embeddings_tsne = tsne.fit_transform(all_embeddings)
    
    out_embeddings_tsne = all_embeddings_tsne[labels==0]
    in_embeddings_tsne = all_embeddings_tsne[labels==1]
   

    kde_in = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(in_embeddings_tsne)
    kde_out = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(out_embeddings_tsne)

# Generate grid for evaluation
    x = np.linspace(min(all_embeddings_tsne[:, 0]), max(all_embeddings_tsne[:, 0]), 100)
    y = np.linspace(min(all_embeddings_tsne[:, 1]), max(all_embeddings_tsne[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack((X.ravel(), Y.ravel())).T

    pdf_in = np.exp(kde_in.score_samples(xy))
    pdf_out = np.exp(kde_out.score_samples(xy))
    kld = compute_kl_divergence(pdf_in, pdf_out)
    return kld





    

