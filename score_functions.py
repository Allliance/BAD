from BAD.eval.eval import evaluate
import torch
from BAD.utils import cosine_similaruty

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_auc(model, dataloader, attack=None, progress=False):
    '''
    This function calculates the AUC of the model on the given dataloader.
    model: the model to be evaluated
    dataloader: the dataloader that contains the data to be evaluated
    attack: an attack function that takes the data and the label as input and returns the adversarial example
    '''
    
    return evaluate(model=model, loader=dataloader, device=device, attack=attack, metric='auc', progress=progress)


def get_aucs(model_dataset, eps, progress=False, verbose=False, score_progress=False):
    attack_eps = eps
    attack_steps = 10
    attack_alpha = 2.5 * attack_eps / attack_steps
    
    def score_function(model, progress=score_progress):
        if verbose:
            print(f"Current model: {model.meta_data['name']}")
        dataloader = get_dataloader(model)
        if score != 'final_auc':
            init_perf = get_auc(model, dataloader, attack=None, progress=progress)
        
        attack = Attack(model, eps=attack_eps, steps=attack_steps, alpha=attack_alpha, attack_in=attack_in)
        if eps == 0:
            attack = None
        adv_perf = get_auc(model, dataloader, attack, progress=progress)
        
        if score == 'final_auc':
            return adv_perf
        elif score == 'prop_auc':
            return adv_perf / init_perf
        elif score == 'delta_auc':
            return init_perf - adv_perf

    return get_models_scores(model_dataset, score_function, progress=progress)


def get_l2(model, dataloader, attack=None, use_in=True, progress=False, normalize_features=False):
    
    mean_initial_features = get_features_mean_dict(dataloader,
                                                   feature_extractor=lambda data, targets: model.get_features(data, normalize_features),
                                                   progress=progress)
    mean_in_initial_features = mean_initial_features[1]
    mean_out_initial_features = mean_initial_features[0]

    initial_diff = (mean_out_initial_features - mean_in_initial_features)
    
    def get_adv_feature_extractor(attack):
        return lambda data, targets : model.get_features(attack(data, targets), normalize_features)
    
    mean_adv_features = get_features_mean_dict(dataloader, get_adv_feature_extractor(attack), progress=progress)
    mean_in_adv_features = mean_adv_features[1]
    mean_out_adv_features = mean_adv_features[0]
    
    if use_in:
        adv_diff = (mean_out_adv_features - mean_in_adv_features)
        score1 = norm(adv_diff - initial_diff)
        
        # score2 = cosine_similaruty(adv_diff, initial_diff)
        # return score1, score2
        
        score = score1
        
        return score
    else:
        diff = mean_out_adv_features - mean_out_initial_features
        score = norm(diff)
        return score

