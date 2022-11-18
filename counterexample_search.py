import pickle

from aalpy.utils import load_automaton_from_file

with open('trace_tree_.pickle', 'rb') as handle:
    print('Cache initialized')
    cache = pickle.load(handle)

current_hyp = load_automaton_from_file('learned_models/lr_8.dot', 'smm')
cex = cache.find_cex_in_cache(current_hyp)
print(cex)