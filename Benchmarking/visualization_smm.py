import csv
import os
from collections import defaultdict
from statistics import mean

directory = 'FM_mdp_smm_error_based_stop/benchmark_no_cq_bfs_longest_prefix/'

benchmarks = os.listdir(directory)

values = dict()

for file in benchmarks:
    with open(directory + file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

        for i in range(0, len(data), 3):
            header = data[i]
            mdp,smm = data[i+1], data[i + 2]

            for formalism in [mdp, smm]:
                for i, val in enumerate(formalism[1:]):
                    if formalism[0] not in values.keys():
                        values[formalism[0]] = defaultdict(list)
                    values[formalism[0]][header[i+1]].append(float(val))

min_values_dict = dict()
max_values_dict = dict()
avr_values_dict = dict()

for exp in values:
    exp_name = exp[12:]
    formalism = 'smm' if 'smm' in exp else 'mdp'

    name = f'{exp_name}_{formalism}'
    min_values_dict[name] = dict()
    max_values_dict[name] = dict()
    avr_values_dict[name] = dict()

    for category, value in values[exp].items():
        min_values_dict[name][category] = min(value)
        max_values_dict[name][category] = max(value)
        avr_values_dict[name][category] = mean(value)

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

interesting_fields = [' Learning time', ' Learning Rounds', ' #MQ Learning', ' # Steps Learning']

experiments = list(min_values_dict.keys())
for e_index in range(0, len(experiments), 2):
    for i in interesting_fields:
        print(f'{experiments[e_index]} vs {experiments[e_index + 1]} = > {i}')
        print(f'Min : {min_values_dict[experiments[e_index]][i]} vs {min_values_dict[experiments[e_index + 1]][i]}')
        print(f'Max : {max_values_dict[experiments[e_index]][i]} vs {max_values_dict[experiments[e_index + 1]][i]}')
        print(f'Avr : {avr_values_dict[experiments[e_index]][i]} vs {avr_values_dict[experiments[e_index + 1]][i]}')
    print('-------------------------------------------------')