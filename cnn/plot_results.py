import os
import sys
import json

import matplotlib.pyplot as plt
import nasbench301 as nb

from genotypes import Genotype

def parse(logpath):
    results = {
        'train': [],
        'valid': [],
        'epoch': [],
        'genotype': [],
    }

    with open(logpath, 'r') as logfile:
        for line in logfile.readlines():
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

    return results


def main(args):
    path1 = os.path.abspath(sys.argv[1])
    path2 = os.path.abspath(sys.argv[2])
    dirname1 = os.path.dirname(path1)
    dirname2 = os.path.dirname(path2)

    resultspath = os.path.join(dirname1, 'results.json')

    results1 = parse(path1)
    results2 = parse(path2)

    print('Loading nasbench models')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'nb_models')
    models = {
        'xgb_performance': os.path.join(model_dir, 'xgb_v1.0'),
        'gnngin_performance': os.path.join(model_dir, 'gnn_gin_v1.0'),
        'lgb_runtime': os.path.join(model_dir, 'lgb_runtime_v1.0')
    }
    xgb_performance = nb.load_ensemble(models['xgb_performance'])
    gnngin_performance = nb.load_ensemble(models['gnngin_performance'])
    lgb_runtime = nb.load_ensemble(models['lgb_runtime'])

    results1['xgb_performance'] = [
        xgb_performance.predict(config=genome, representation='genotype',
                                with_noise=True)
        for genome in results1['genotype']
    ]
    results1['gnngin_performance'] = [
        gnngin_performance.predict(config=genome, representation='genotype',
                                with_noise=True)
        for genome in results1['genotype']
    ]
    results1['lgb_runtime'] = [
        lgb_runtime.predict(config=genome, representation='genotype',
                                with_noise=True)
        for genome in results1['genotype']
    ]

    results2['xgb_performance'] = [
        xgb_performance.predict(config=genome, representation='genotype',
                                with_noise=True)
        for genome in results2['genotype']
    ]
    results2['gnngin_performance'] = [
        gnngin_performance.predict(config=genome, representation='genotype',
                                with_noise=True)
        for genome in results2['genotype']
    ]
    results2['lgb_runtime'] = [
        lgb_runtime.predict(config=genome, representation='genotype',
                                with_noise=True)
        for genome in results2['genotype']
    ]

    with open(resultspath, 'w') as resultsfile:
        results1['genotype'] = list(map(str, results1['genotype']))
        results2['genotype'] = list(map(str, results2['genotype']))
        results = {
            'darts': results1,
            'fft': results2
        }
        json.dump(results, resultsfile)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for t in ['darts', 'fft']:
        ax1.plot(results[t]['epoch'], results[t]['train'], label=f'{t}_train')
        ax1.plot(results[t]['epoch'], results[t]['valid'], label=f'{t}_valid')
        ax2.plot(results[t]['epoch'], results[t]['xgb_performance'], label=f'{t}_xgb')
        ax2.plot(results[t]['epoch'], results[t]['gnngin_performance'], label=f'{t}_gnngin')

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'python {sys.argv[0]} path/to/dartslog.txt path/to/fftlog.txt')
        sys.exit(1)

    main(sys.argv)
