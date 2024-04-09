
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