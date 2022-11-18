import sys
from random import choice, choices, randint

from BLESUL import BLESUL
from FailSafeOracle import FailSafeOracle
from aalpy.automata import Onfsm
from aalpy.base import SUL
from aalpy.learning_algs import run_non_det_Lstar, run_Alergia
# from FailSafeOracle import FailSafeOracle
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle
from aalpy.utils import load_automaton_from_file
from aalpy.learning_algs import run_stochastic_Lstar
from FailSafeLearning.StatePrefixEqOracleFailSafe import StatePrefixOracleFailSafe

# def test_counter():
serial_port = sys.argv[1]
advertiser_address = sys.argv[2]

model = load_automaton_from_file("correct_model.dot", "onfsm")
alphabet = model.get_input_alphabet()
print(alphabet)
sul = BLESUL(serial_port, advertiser_address)

model = load_automaton_from_file("correct_model.dot", "onfsm")

# eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=1000, reset_prob=0.05, reset_after_cex=True)
eq_oracle = StatePrefixOracleFailSafe(alphabet, sul, walks_per_state=100, walk_len=20)

counterexample = eq_oracle.find_cex(model)
if counterexample is not None:
    print(counterexample)
    exit()
# eq_oracle = FailSafeOracle(alphabet, sul, num_walks=1000, min_walk_len=4, max_walk_len=10, reset_after_cex=False)
# eq_oracle = RandomWordEqOracle(alphabet, sul)

# learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, debug=True, stochastic="smm")

# eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=10, reset_after_cex=False)

# learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, print_level=2, pruning_threshold=0.2, debug=True, stochastic="smm")
# learned_model = run_stochastic_Lstar(alphabet, sul, eq_oracle, min_rounds=10, automaton_type="smm")


        # learned_model.visualize()

