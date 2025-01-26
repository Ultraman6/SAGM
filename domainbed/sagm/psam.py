import torch
from .util import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp


class PSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, alpha, rho_scheduler=None, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(PSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = alpha
        if self.rho_scheduler:
            self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["f"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product

        inner_prod_hf_p = 0.0
        inner_prod_hf_P_f = 0.0
        inner_prod_hf_P_h = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['h'] = p.grad.data - self.state[p]['f']
                self.state[p]['hf'] = self.state[p]['h'] - self.state[p]['f']
                inner_prod_hf_p += torch.sum(p.grad.data * self.state[p]['hf'])
                # inner_prod_f_hf += torch.sum(self.state[p]['f_g'] * self.state[p]['hf_g'])
                # inner_prod_h_hf += torch.sum(self.state[p]['h_g'] * self.state[p]['hf_g'])

        # sign_p_hf = -1 if inner_prod_p_hf > 0 else -1
        # sign_f_hf = -1 if inner_prod_f_hf > 0 else -1
        # sign_h_hf = -1 if inner_prod_h_hf > 0 else -1

        # get norm 1
        grad_norm_p = self._grad_norm()
        grad_norm_hf = self._grad_norm(by='hf')
        # get cosine 1
        cosine_hf_p = inner_prod_hf_p / (grad_norm_p * grad_norm_hf + self.perturb_eps)
        # gradient decomposition 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['hf_P'] = cosine_hf_p * grad_norm_hf * p.grad.data / (grad_norm_p + self.perturb_eps)
                hf_p_V = self.state[p]['hf'] - self.state[p]['hf_P']
                sign_hf_p = torch.sign(torch.sum(p.grad.data * self.state[p]['hf']))
                p.grad.data.add_(hf_p_V, alpha=sign_hf_p * alpha)
                inner_prod_hf_P_f += torch.sum(self.state[p]['hf_P'] * self.state[p]['f'])
                inner_prod_hf_P_h += torch.sum(self.state[p]['hf_P'] * self.state[p]['h'])
        # get norm 2
        grad_norm_f = self._grad_norm(by='f')
        grad_norm_h = self._grad_norm(by='h')
        grad_norm_hf_P = self._grad_norm(by='hf_p')
        # get cosine 2
        cosine_hf_P_f = inner_prod_hf_P_f / (grad_norm_hf_P * grad_norm_f + self.perturb_eps)
        cosine_hf_P_h = inner_prod_hf_P_h / (grad_norm_hf_P * grad_norm_h + self.perturb_eps)
        # gradient decomposition 2
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                hf_P_f_V = self.state[p]['hf_P'] - cosine_hf_P_f * grad_norm_hf_P * self.state[p]['f'] / (grad_norm_f + self.perturb_eps)
                hf_P_h_V = self.state[p]['hf_P'] - cosine_hf_P_h * grad_norm_hf_P * self.state[p]['h'] / (grad_norm_h + self.perturb_eps)
                sign_hf_P_f = torch.sign(torch.sum(self.state[p]['hf_P'] * self.state[p]['f']))
                sign_hf_P_h = torch.sign(torch.sum(self.state[p]['hf_P'] * self.state[p]['h']))
                p.grad.data.add_(hf_P_f_V, alpha=sign_hf_P_f * alpha)
                p.grad.data.add_(hf_P_h_V, alpha=sign_hf_P_h * alpha)


    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value
