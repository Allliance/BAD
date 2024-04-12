import torch
from BAD.utils.visualization import plot_process

def find_eps_upperbound(evaluator, thresh=0.4, log=False):
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

def find_best_gap(m1, m2, evaluator, config, log=False):
    
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
            evaluator(m1, attack=attack_class(m1, **(get_attack_params(eps) | config['attack_params']))), log=log)
        
    eps_steps = config.get('eps_steps')
    if eps_steps is None:
        eps_steps = 10
    
    
    epsilons = torch.linspace(eps_lb, eps_ub, eps_steps).tolist()
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