import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from numpy.linalg import norm
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_features(model, loader, DEVICE, attack=None, progress=False):
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
        data, label = data.to(DEVICE), label.to(DEVICE)
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
def get_max_diff(model,testloader, attack_config=None, score='l2', use_in=True, progress=False):
    max_l2 = 0
    tq = range(10)
    if progress:
        tq = tqdm(range(10))
    
    attack_class = attack_config['attack_class']
    attack_config.pop('attack_class')
    
    for i in tq:
        attack_config['target_class'] = i
        attack = attack_class(model, **attack_config)
        v_out_b, v_in_b= get_features(model, testloader, device, attack=None)
        v_out_a, v_in_a = get_features(model, testloader, device, attack=attack)
        v_out_b_mean = np.mean(v_out_b, axis=0)
        v_out_a_mean = np.mean(v_out_a, axis=0)
        v_in_b_mean = np.mean(v_in_b, axis=0)
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
                           
    return max_l2


def visualize_samples(dataloader, n, title="Sample"):
    plt.clf()
    normal_samples = []
    abnormal_samples = []

    def to_3_channels(image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        return image

    # Collect n x n samples
    for images, labels in dataloader:
        for i, l in enumerate(labels):
            image = to_3_channels(images[i])
            if len(normal_samples) < n * n and l == 0:
                normal_samples.append(image)
            elif len(abnormal_samples) < n * n and l != 0:
                abnormal_samples.append(image)
            if len(normal_samples) == n * n and len(abnormal_samples) == n * n:
                break
        if len(normal_samples) == n * n and len(abnormal_samples) == n * n:
            break

    
    normal_grid = torchvision.utils.make_grid(normal_samples, nrow=n)
    abnormal_grid = torchvision.utils.make_grid(abnormal_samples, nrow=n)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    fig.patch.set_alpha(0)
    fig.suptitle(title, fontsize=16)

    axs[0].imshow(normal_grid.permute(1, 2, 0))
    axs[0].set_title('Normal', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(abnormal_grid.permute(1, 2, 0))
    axs[1].set_title('Abnormal', fontsize=14)
    axs[1].axis('off')

    plt.show()
    

def plot_gaps(cleans, bads, dataset, best_eps, verbose=False):
    
    plt.clf()
    # Plot both arrays
    x = np.arange(len(cleans))  # the label locations

    width = 0.35  
    
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    for i in range(len(cleans)):
        ax.text(x[i], -0.1, f"Eps: {best_eps[i]:.3f}", ha='center')
        
    rects1 = ax.bar(x - width/2, cleans, width, label='clean resnet', color='blue')
    rects2 = ax.bar(x + width/2, bads, width, label='bad resnet', color='red')

    ax.set_ylabel('auroc')
    ax.set_title(f'Bar Chart for {dataset} for best eps')
    ax.set_xticks(x)
    ax.legend()

    fig.savefig(f'results/{dataset}.png', bbox_inches='tight')
    if verbose:
        plt.show()


def plot_process(epsilons, gaps, title, verbose=False):
    plt.clf()
    
    plt.plot(epsilons, gaps)  # Plot y1 with blue color
    plt.xlabel('epsilon')
    plt.ylabel('auroc gap')
    plt.title(title)
    plt.savefig(f'results/{title}.png')
    if verbose:
        plt.show()


def ood_sanity_check(testloader, adv_check=True, clean_check=True, target=None):
    print("Sanity check started")
        
    # Sanity Check for Clean OOD Detection
    if clean_check:
        print("Performing sanity check for clean ood detection performance")
        clean_roc = get_ood(clean_model, testloader, target=target, attack_eps=0, progress=DEBUG)
        bd_roc = get_ood(bad_model, testloader, target=target, attack_eps=0, progress=DEBUG)
        print("Clean auroc with ood detection:", clean_roc)
        print("BD auroc on ood detection:", bd_roc)
    
    # Sanity Check for Adversarial Attack
    if adv_check:
        print("Performing sanity check for adversarial attack")
        clean_roc = get_ood(clean_model, testloader, target=target, attack_eps=32/255, progress=DEBUG)
        bd_roc = get_ood(bad_model, testloader, target=target, attack_eps=32/255, progress=DEBUG)
        print("Clean auroc with large epsilon:", clean_roc)
        print("BD auroc with large epsilon:", bd_roc)
        
    print("End of Sanity Check")

def find_eps_upperbound(evaluator, thresh, log=False):
    for j in range(1, 32):
        score = evaluator(j/255)
        if score < thresh:
            upper_attack_eps = j/255
            if log:
                print("Thresh obtained at epsilon =",f"{j}/255, with score=", score)
            break
    else:
        upper_attack_eps = 10/255
        if log:
            print("No good eps found")
        
    return upper_attack_eps

def get_attack_params(attack_eps=8/255, attack_steps=10):
    attack_alpha = 2.5 * attack_eps / attack_steps
    
    return {
        'eps': attack_eps,
        'alpha': attack_alpha,
        'steps': attack_steps
    }

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