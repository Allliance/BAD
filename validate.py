import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Currently only implemented for Auc on Auc
def find_best_config(model_set, dataset, num_classes,
                     score_function, score_function_params, config_search_space):
    pass
