from random import seed

from aalpy.SULs import OnfsmSUL
from aalpy.learning_algs import run_Lstar_ONFSM
from aalpy.oracles import UnseenOutputRandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, visualize_automaton

seed(12)

mqtt_client = load_automaton_from_file('DotModels/mqtt_multi_client_solution.dot', automaton_type='mealy')

alphabet = mqtt_client.get_input_alphabet()

sul = OnfsmSUL(mqtt_client)
eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)

learned_model = run_Lstar_ONFSM(alphabet, sul, eq_oracle=eq_oracle, n_sampling=5)