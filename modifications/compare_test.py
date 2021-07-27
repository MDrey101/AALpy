from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_stochastic_Lstar
import PropertyDirectedSamplingStormpy as PDSS
from aalpy.utils import load_automaton_from_file
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

original_mdp = load_automaton_from_file("/home/manuel/TUGraz/SPST/AALpy/DotModels/MDPs/first_grid.dot",
                                        automaton_type='mdp')
input_alphabet = original_mdp.get_input_alphabet()
mdp_sul = MdpSUL(original_mdp)

random_oracle = PDSS.PDS(target="goal", sul=mdp_sul, alphabet=input_alphabet, k=10, quit_prob=0.025, n_batch=250,
                         p_rand=1, c_change=1, stop_on_cex=False)
pds_oracle = PDSS.PDS(target="goal", sul=mdp_sul, alphabet=input_alphabet, k=10, quit_prob=0.025, n_batch=250,
                      p_rand=0.9, stop_on_cex=False)

rand_acc = []
pds_acc = []
for x in range(10):
    random_Lstar_mdp, random_data = run_stochastic_Lstar(input_alphabet=input_alphabet, sul=mdp_sul,
                                                         eq_oracle=random_oracle, n_resample=125, max_rounds=100,
                                                         samples_cex_strategy=None, return_data=True, evaluation=True)
    pds_Lstar_mdp, pds_data = run_stochastic_Lstar(input_alphabet=input_alphabet, sul=mdp_sul, eq_oracle=pds_oracle,
                                                   n_resample=125, max_rounds=100, samples_cex_strategy=None,
                                                   return_data=True, evaluation=True)

    rand_acc.append(random_oracle.accuracy_list.copy())
    random_oracle.accuracy_list = []
    pds_acc.append(pds_oracle.accuracy_list.copy())
    pds_oracle.accuracy_list = []

# ----------------------------------------------------------------------------------------------------------------------

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
plt.ylabel("F<10 goal")
plt.title("random, first_grid, k=10")
plt.show()
# plt.savefig("random.png")
# tikzplotlib.save("test1_3.tex")

plt.plot(list(range(len(avg_pds_acc))), avg_pds_acc, "b")
plt.plot(list(range(len(min_pds_acc))), min_pds_acc, "g")
plt.plot(list(range(len(max_pds_acc))), max_pds_acc, "r")
plt.plot(list(range(len(third_quantile_pds_acc))), third_quantile_pds_acc, "c")
plt.plot(list(range(len(first_quantile_pds_acc))), first_quantile_pds_acc, "m")
plt.xlabel("# rounds")
plt.ylabel("F<10 goal")
plt.title("pds, first_grid, k=10")
plt.show()
# plt.savefig("pds.png")
# tikzplotlib.save("test2_3.tex")
