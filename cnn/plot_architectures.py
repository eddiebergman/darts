import os
import re
import sys
import json
import numpy

import matplotlib.pyplot as plt

from genotypes import Genotype

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

def parse(logpath):
    results = {
        'train': [],
        'valid': [],
        'epoch': [],
        'genotype': [],
        'alphas_normal_softmax': [],
        'alphas_reduce_softmax': [],
        'alphas_normal': [],
        'alphas_reduce': [],
    }

    with open(logpath, 'r') as logfile:
        lines = list(logfile.readlines())
        for i, line in enumerate(lines):
            # Spaces neccessary
            if ' train_acc ' in line:
                number = float(line.split()[-1])
                results['train'].append(number)

            elif ' valid_acc ' in line:
                number = float(line.split()[-1])
                results['valid'].append(number)

            elif ' epoch ' in line:
                number = int(line.split()[-1])
                results['epoch'].append(number)

            elif ' genotype ' in line:
                genome_str = line[line.find("Genotype"):]
                genome = eval(genome_str)
                results['genotype'].append(genome)

            elif ' alphas:' in line:
                num_lines = 15 if 'softmax' in line else 29
                tensor_lines = lines[i:i + num_lines]
                s = ''.join(tensor_lines)
                s = s[s.index('tensor'):] # from tensor onwards
                s = re.sub('\n|\r|\s+', '', s)

                # Probably a nicer way to do this with regex
                s = s.replace('tensor(', '')
                s = s.replace(',requires_grad=True', '')
                s = s.replace(',requires_grad=False', '')
                s = s.replace(',device=\'cuda:0\'', '')
                s = s.replace(',grad_fn=<SoftmaxBackward>', '')
                # Remove the bracket from the last line
                s = s[:-1]

                arr = numpy.asarray(eval(s))
                if 'Normal alphas' in line:
                    results['alphas_normal'].append(arr)
                elif 'Reduce alphas' in line:
                    results['alphas_reduce'].append(arr)
                elif 'Normal softmax alphas' in line:
                    results['alphas_normal_softmax'].append(arr)
                elif 'Reduce softmax alphas' in line:
                    results['alphas_reduce_softmax'].append(arr)

                print(results)

    return results

def chosen_ops(alphas):
    nodes = 4
    gene = []
    n = 2
    start = 0
    for node in range(nodes):
        end = start + n
        W = weights[start: start + n].copy()
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


def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(sys.argv[1])
    dirname = os.path.dirname(path)

    figpath = os.path.join(dirname, 'architectures_plot.png')


    results = parse(path)
    epochs = len(results['alphas_normal'])
    fig, axs = plt.subplots(epochs, 4)
    axs = axs.flatten()
    y_ticks = [str((i, j) for i in range(j) for j in range(2, 4))]
    for epoch in range(epochs):
        axs[epoch*4 + 0].imshow(results['alphas_normal'][epoch], cmap='gray')
        axs[epoch*4 + 0].set_title('Normal Cell')
        axs[epoch*4 + 0].yticks(range(14), [y_kkk])

        axs[epoch*4 + 1].imshow(results['alphas_normal_softmax'][epoch], cmap='gray')
        axs[epoch*4 + 1].set_title('Normal Cell (Softmax)')

        axs[epoch*4 + 2].imshow(results['alphas_reduce'][epoch], cmap='gray')
        axs[epoch*4 + 2].set_title('Reduce Cell')

        axs[epoch*4 + 3].imshow(results['alphas_reduce_softmax'][epoch], cmap='gray')
        axs[epoch*4 + 3].set_title('Reduce Cell (Softmax)')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'python {sys.argv[0]} path/to/log.txt')
        sys.exit(1)

    main(sys.argv)
