import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev,
                 C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion,
                 steps=4, multiplier=4, stem_multiplier=3,
                 retain_arch_grad=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._retain_arch_grad = retain_arch_grad

        # Set up first layer
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # Set up intermediate layers
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False

        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Create architecture parameters
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self._arch_parameters = {
            'normal': 1e-3 * torch.randn(k, num_ops).cuda(),
            'reduce': 1e-3 * torch.randn(k, num_ops).cuda()
        }

        # Set them to leaf nodes that require grads
        for tensor in self._arch_parameters.values():
            tensor.requires_grad = True

    def clone(self):
        """ Clones the architecture

        Note: This will not clone any weights
        """
        new_model = Network(self._C,
                            self._num_classes,
                            self._layers,
                            self._criterion)
        new_model.cuda()

        # Transfer arch parameters to new model
        local_arch = self._arch_parameters
        new_model_arch = new_model._arch_parameters
        for key in ['normal', 'reduce']:
            local_weights = local_arch[key]
            new_model_weights = new_model_arch[key]
            new_model_weights.copy_(local_weights)

        return new_model

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce(), dim=-1)
            else:
                weights = F.softmax(self.alphas_normal(), dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def arch_parameters(self):
        return self._arch_parameters

    def alphas_normal(self):
        return self._arch_parameters['normal']

    def alphas_reduce(self):
        return self._arch_parameters['reduce']

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -
                               max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(
                self.alphas_normal(),
                dim=-
                1).data.cpu().numpy())
        gene_reduce = _parse(
            F.softmax(
                self.alphas_reduce(),
                dim=-
                1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


class FFTNetwork(Network):

    def __init__(self, C, num_classes, layers, criterion,
                 steps=4, multiplier=4, stem_multiplier=3,
                 retain_arch_grad=False, coeff_size=(14, 8)):
        super().__init__(C, num_classes, layers, criterion, steps,
                         multiplier, stem_multiplier, retain_arch_grad)
        self._coeff_size = coeff_size

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        init_alphas_normal = 1e-3 * torch.randn(k, num_ops).cuda()
        init_alphas_reduce = 1e-3 * torch.randn(k, num_ops).cuda()

        self._arch_sizes = {
            'normal': init_alphas_normal.size(),
            'reduce': init_alphas_reduce.size()
        }

        # Note: We store the full frequency spectrum but this gets
        # trimmed for backprop, alphas reconstruction and arch_coefficients()
        self._coeffs = {
            'normal': fft.rfft2(init_alphas_normal, s=self._arch_sizes['normal']),
            'reduce': fft.rfft2(init_alphas_reduce, s=self._arch_sizes['reduce'])
        }

        # Set grad for _coeffs
        for tensor in self._coeffs.values():
            tensor.requires_grad = True

    def clone(self):
        """ Clones the architecture

        Note: This will not clone any weights
        """
        new_model = Network(self._C,
                            self._num_classes,
                            self._layers,
                            self._criterion)
        new_model.cuda()

        # Transfer arch parameters to new model
        local_arch = self._arch_parameters
        new_model_arch = new_model._arch_parameters
        for key in ['normal', 'reduce']:
            local_weights = local_arch[key]
            new_model_weights = new_model_arch[key]
            new_model_weights.copy_(local_weights)

        return new_model

        return

    def alphas_normal(self):
        coeffs = self.arch_coefficients()
        alphas = fft.irfft2(coeffs['normal'], s=self._arch_sizes['normal'])
        if self._retain_arch_grad:
            alphas.retain_grad()

        return alphas

    def alphas_reduce(self):
        coeffs = self.arch_coefficients()
        alphas = fft.irfft2(coeffs['reduce'], s=self._arch_sizes['reduce'])
        if self._retain_arch_grad:
            alphas.retain_grad()

        return alphas

    def arch_parameters(self):
        return {
            'normal': self.alphas_normal(),
            'reduce': self.alphas_reduce()
        }

    def arch_coefficients(self, pruned=True):
        if not pruned:
            return self._coeffs
        else:
            # Trims the higher frequency compenents based on self._coeff_size
            rows, cols = self._coeff_size
            return {
                'normal': self._coeffs['normal'][:rows, :cols],
                'reduce': self._coeffs['reduce'][:rows, :cols]
            }
