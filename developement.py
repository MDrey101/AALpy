from random import seed

from aalpy.SULs import OnfsmSUL
from aalpy.learning_algs import run_Lstar_ONFSM
from aalpy.oracles import UnseenOutputRandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, visualize_automaton

seed(12)

onfsm_example = load_automaton_from_file('DotModels/onfsm_2.dot', automaton_type='onfsm')

alphabet = onfsm_example.get_input_alphabet()

sul = OnfsmSUL(onfsm_example)
eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)

learned_model = run_Lstar_ONFSM(alphabet, sul, eq_oracle=eq_oracle, n_sampling=25)

visualize_automaton(learned_model,path="onfsm_2",file_type='dot')