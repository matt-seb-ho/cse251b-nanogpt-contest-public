import torch

def newtonschulz(G, steps=5):
    """Newton-Schulz iteration to orthogonalize G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr = 0.02, momentum = 0.95, weight_decay = 0, ns_steps = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay = weight_decay, ns_steps = ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buffer = state['momentum_buffer']
                buffer.mul_(group['momentum']).add_(g)
                g = g.add(buffer, alpha=group['momentum'])
                update = newtonschulz(g, steps=group['ns_steps'])
                update = update.to(p.dtype)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape), alpha=-group['lr'])