def find_best_eps(m1, m2, dataloader, device, thresh=0.4, eps_steps=20):
    thresh = 0.55
        for j in range(1, 17):
                attack_eps = j/255
                attack_steps = 10
                attack_alpha = 2.5 * attack_eps / attack_steps
                clean_roc = get_ood(clean_model, testloader, device)
                if clean_roc < thresh:
                    upper_attack_eps = j/255
                    print("Thresh obtained for target", i, "at epsilon =",f"{j}/255, with auroc=", clean_roc)
                    break
        else:
                upper_attack_eps = 8/255
                print("No good eps found for target", i)
                print("Use 8/255 for eps for this target")
        best_gap = -20
        epsilons = torch.linspace(0, upper_attack_eps, 2 if DEBUG else 20 * int(255*upper_attack_eps)) 
        eps = []
        gaps = []
        
    
        for attack_eps in epsilons:
            if DEBUG:
                print("Working on epsilon", attack_eps.item() * 255)
        
            torch.cuda.empty_cache()
            gc.collect()
        
            attack_eps = attack_eps.item()
            attack_steps = 10
            attack_alpha = 2.5 * attack_eps / attack_steps

            roc1 = get_ood(clean_model, testloader, device, DEBUG)
            roc2 = get_ood(bad_model, testloader, device, DEBUG)
            gaps.append(roc1-roc2)
            eps.append(attack_eps * 255)
            if DO_LOGGING:
                print(f'clean model auroc under adversarial attack: {roc1}')
                print(f'bad model auroc under adversarial attack: {roc2}')

            if roc1-roc2 > best_gap:
                best_gap = roc1-roc2
                best_epses[i] = attack_eps*255
                cleans[i] = roc1
                bads[i] = roc2
                
                print(dataset, "--- Best gap until eps =", attack_eps * 255, "is", best_gap, f'target is {i}')
            
        
        plot_process(eps,gaps,dataset,i)
    plot_gaps(cleans, bads, dataset, best_epses)