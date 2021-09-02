from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_stochastic_Lstar
import PropertyDirectedSampling as PDS
from aalpy.utils import load_automaton_from_file
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import re
import tikzplotlib

example_path = os.path.join(Path(__file__).parent.parent, "DotModels", "MDPs")
cache_path = os.path.join(os.getcwd(), "cache")
example_setups = [
    {"example": "first_grid", "target": "goal", "steps": 10},
    {"example": "slot_machine", "target": "Pr10", "steps": 14},
    {"example": "shared_coin", "target": "finished", "steps": 20}
]

example_setup = example_setups[1]
use_tikzplot = False


def check_for_data(path, example):
    """
    Check for cached data; the corresponding cache text is checked if it contains values in list entries

    Args:
        path: path to cache directory containing logs
        example: actual logfile

    Returns: 2 lists with values for random and pds accuracies

    """
    rand_acc = []
    pds_acc = []

    selector = True
    temp_rand_acc = []
    temp_pds_acc = []
    try:
        with open(os.path.join(path, example + ".txt"), "r") as infile:
            for line in infile:
                if re.match("(---)+", line):
                    if selector:
                        rand_acc.append(temp_rand_acc.copy())
                        temp_rand_acc = []
                    else:
                        pds_acc.append(temp_pds_acc.copy())
                        temp_pds_acc = []
                    selector = not selector
                elif selector:
                    temp_rand_acc.append(float(line))
                else:
                    temp_pds_acc.append(float(line))
        print(rand_acc)
        print(pds_acc)
        return rand_acc, pds_acc
    except FileNotFoundError:
        print("No cache data found")
    except Exception:
        print(f"Something went wrong when checking for cache data for {example}.txt")
    return [], []


def random_pds_compare_test(random_cache_data, pds_cache_data, path, example, target, steps):
    """
    Creates a random and a pds oracle and executes both in a loop with 10 rounds to create a dataset used for plotting

    Args:
        random_cache_data: accuracy values found for random setup
        pds_cache_data: accuracy values found for pds setup
        path: path to directory containing logs
        example: example file to be executed
        target: target state for the oracle
        steps: upper bound of steps for the oracle

    Returns: two lists containing all accuracy values

    """
    original_mdp = load_automaton_from_file(os.path.join(example_path, example + ".dot"),
                                            automaton_type='mdp')
    input_alphabet = original_mdp.get_input_alphabet()
    mdp_sul = MdpSUL(original_mdp)

    random_oracle = PDS.PDS(target=target, sul=mdp_sul, alphabet=input_alphabet, k=steps, quit_prob=0.025, n_batch=250,
                             p_rand=1, c_change=1, stop_on_cex=False)
    pds_oracle = PDS.PDS(target=target, sul=mdp_sul, alphabet=input_alphabet, k=steps, quit_prob=0.025, n_batch=250,
                          p_rand=0.9, stop_on_cex=False)

    rand_acc = random_cache_data
    pds_acc = pds_cache_data
    if len(rand_acc) > len(pds_acc):
        execute_lstar(input_alphabet, mdp_sul, pds_oracle)
        write_output(pds_oracle, path, example)

    for x in range(max(len(pds_acc), len(rand_acc)), 10):
        execute_lstar(input_alphabet, mdp_sul, random_oracle)
        write_output(random_oracle, path, example)

        execute_lstar(input_alphabet, mdp_sul, pds_oracle)
        write_output(pds_oracle, path, example)

        rand_acc.append(random_oracle.accuracy_list.copy())
        pds_acc.append(pds_oracle.accuracy_list.copy())
        random_oracle.accuracy_list = []
        pds_oracle.accuracy_list = []

    return rand_acc, pds_acc


def write_output(oracle, path, example):
    """

    Args:
        oracle: Oracle used to get accuracy list values
        path: path to directory containing logs
        example: example file to write to

    """
    with open(os.path.join(path, example + ".txt"), "a") as outfile:
        for line in oracle.accuracy_list:
            outfile.write(str(line) + "\n")
        outfile.write("-" * 35 + "\n")


def execute_lstar(input_alphabet, mdp_sul, oracle):
    """

    Args:
        input_alphabet: input alphabet of the SUL
        mdp_sul: SUL that is handed over to LStar
        oracle: oracle used by LStar

    """
    run_stochastic_Lstar(input_alphabet=input_alphabet, sul=mdp_sul,
                         eq_oracle=oracle, n_resample=125, max_rounds=100,
                         samples_cex_strategy=None, return_data=True, evaluation=True)


def plot_data(rand_acc, pds_acc, example):
    """
    Creates a random and a pds plot for the calculated data

    Args:
        rand_acc: list with random computed accuracies
        pds_acc: list with pds computed accuracies
        example: example setup including name, target and steps

    """
    max_len_entry = 0
    for entry in rand_acc:
        if len(entry) > max_len_entry:
            max_len_entry = len(entry)

    for entry in rand_acc:
        for e in range(max_len_entry - len(entry) + 1):
            entry.append(entry[-1])

    max_len_entry = 0
    for entry in pds_acc:
        if len(entry) > max_len_entry:
            max_len_entry = len(entry)

    for entry in pds_acc:
        for e in range(max_len_entry - len(entry) + 1):
            entry.append(entry[-1])

    numpy_rand_acc = np.array(rand_acc)
    numpy_pds_acc = np.array(pds_acc)

    avg_rand_acc = np.median(numpy_rand_acc, axis=0)
    avg_pds_acc = np.median(numpy_pds_acc, axis=0)

    min_rand_acc = np.min(numpy_rand_acc, axis=0)
    max_rand_acc = np.max(numpy_rand_acc, axis=0)
    min_pds_acc = np.min(numpy_pds_acc, axis=0)
    max_pds_acc = np.max(numpy_pds_acc, axis=0)

    third_quantile_rand_acc = np.quantile(numpy_rand_acc, 0.75, axis=0)
    first_quantile_rand_acc = np.quantile(numpy_rand_acc, 0.25, axis=0)
    third_quantile_pds_acc = np.quantile(numpy_pds_acc, 0.75, axis=0)
    first_quantile_pds_acc = np.quantile(numpy_pds_acc, 0.25, axis=0)

    plt.plot(list(range(len(avg_rand_acc))), avg_rand_acc, "b")
    plt.plot(list(range(len(min_rand_acc))), min_rand_acc, "g")
    plt.plot(list(range(len(max_rand_acc))), max_rand_acc, "r")
    plt.plot(list(range(len(third_quantile_rand_acc))), third_quantile_rand_acc, "c")
    plt.plot(list(range(len(first_quantile_rand_acc))), first_quantile_rand_acc, "m")
    plt.xlabel("# rounds")
    plt.ylabel(f"F<{example['steps']} {example['target']}")
    plt.title(f"random, {example['example']}, k={example['steps']}")
    if use_tikzplot:
        tikzplotlib.save(f"random_{example['example']}.tex")
    else:
        plt.show()

    plt.plot(list(range(len(avg_pds_acc))), avg_pds_acc, "b")
    plt.plot(list(range(len(min_pds_acc))), min_pds_acc, "g")
    plt.plot(list(range(len(max_pds_acc))), max_pds_acc, "r")
    plt.plot(list(range(len(third_quantile_pds_acc))), third_quantile_pds_acc, "c")
    plt.plot(list(range(len(first_quantile_pds_acc))), first_quantile_pds_acc, "m")
    plt.xlabel("# rounds")
    plt.ylabel(f"F<{example['steps']} {example['target']}")
    plt.title(f"pds, {example['example']}, k={example['steps']}")
    if use_tikzplot:
        tikzplotlib.save(f"pds_{example['example']}.tex")
    else:
        plt.show()


if __name__ == "__main__":
    random_cache, pds_cache = check_for_data(cache_path, example_setup["example"])
    random, pds = random_pds_compare_test(random_cache, pds_cache, cache_path, example_setup["example"],
                                          example_setup["target"], example_setup["steps"])
    plot_data(random, pds, example_setup)
