import torch
import torch.nn as nn

from ..attack import Attack

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, target_class=None, eps=8/255, alpha=2/255, steps=10, random_start=True, attack_in=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.target_class = target_class
        self.attack_in = attack_in

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        all_images = images.clone()
        float_labels = labels.clone().detach().type(torch.FloatTensor).to(self.device)
        
        loss = nn.CrossEntropyLoss(reduction='none')
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        hend_multipliers = float_labels
        out_multipliers = 1-float_labels
        
        if not self.attack_in:
            adv_images = adv_images[labels == 0]
            images = images[labels == 0]

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            probs = torch.softmax(outputs, dim=1)
            
            if self.target_class is not None:
                out_loss = -loss(outputs, torch.ones_like(labels) * self.target_class)
            else:
                out_loss = torch.max(probs, dim=1).values
            if self.attack_in:
                cost = torch.dot(out_multipliers, out_loss)
            else:
                cost = torch.dot(torch.ones_like(out_loss), out_loss)
            if self.attack_in:
                hend_loss = 1 * (probs.mean(1) - torch.logsumexp(probs, dim=1))
                cost = cost + torch.dot(hend_multipliers, hend_loss)
            
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if not self.attack_in:
            final_adv_images = all_images.clone()
            final_adv_images[labels == 0] = adv_images

            adv_images = final_adv_images
        return adv_images