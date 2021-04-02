from random import seed

from aalpy.SULs import OnfsmSUL
from aalpy.learning_algs import run_abstracted_Lstar_ONFSM
from aalpy.oracles import UnseenOutputRandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, visualize_automaton

seed(1)

onfsm_example = load_automaton_from_file('DotModels/onfsm_3.dot', automaton_type='onfsm')

alphabet = onfsm_example.get_input_alphabet()

sul = OnfsmSUL(onfsm_example)
eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=10000, reset_prob=0.25, reset_after_cex=True)

abstraction_mapping = dict()

abstraction_mapping[0] = 0
abstraction_mapping['O'] = 0

learned_model = run_abstracted_Lstar_ONFSM(alphabet, sul, eq_oracle=eq_oracle, abstraction_mapping=abstraction_mapping,
                                           n_sampling=100, print_level=3)

visualize_automaton(learned_model, path="abstracted_onfsm_3", file_type='dot')
