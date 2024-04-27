def load_trojai(record_path, num_classes=num_classes, arch=None, round=0, **model_kwargs):
    net = torch.load(record_path, map_location=device)
    
    if arch == 'inceptionv3':
        new_net = inception_v3(num_classes=num_classes)
        new_net.load_state_dict(deepcopy(net.state_dict()), strict=False)
        net = new_net
    
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
    
    model = Model(net, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model