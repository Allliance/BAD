from .eval import eval

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

def get_attack_params(attack_eps):
    return {
        'eps': attack_eps,
        'alpha': attack_alpha,
        'steps': attack_steps
    }

def find_best_gap(m1, m2, evaluator, config, log=False):
    
    print("Working on current config:", config['title'])
    
    best_result = {
        1: 0,
        2: 0,
        'gap': -100,
    }
    
    eps_lb = config.get('eps_lb')
    if eps_lb is None:
        eps_lb = 0
    eps_ub = config.get('eps_ub')
    
    if eps_ub is None:
        eps_ub = find_eps_upperbound(lambda eps: eval(m1, testloader, device,
                                                      attack=config['attack'](m1, target_map=target_map, **get_attack_params(eps)),
                                                      progress=False), log=log)
        
    epsilons = torch.linspace(eps_lb, eps_ub, config['eps_steps']).tolist()
    gaps = []

    for eps in epsilons:
        if log:
            print("Working on epsilon", eps * 255)
    
        torch.cuda.empty_cache()
        gc.collect()
    
        attack = attack_config['attack']
        
        attack_params = get_attack_params(eps)
        
        attack1 = attack(clean_model, target_map=target_map, **attack_params)
        attack2 = attack(bad_model, target_map=target_map, **attack_params)

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
            
            print(f"{config['title']} --- Best gap until eps = {attack_eps * 255} is {best_result['gap']}")    
    
    plot_process(epsilons, gaps, config['title'])
    return best_result