import os
import shutil
import sys
from random import choice, choices, randint

from BLESUL import BLESUL
from FailSafeLearning.FailSafeCacheSUL import FailSafeCacheSUL
from FailSafeLearning.FailSafeSUL import FailSafeSUL
from FailSafeOracle import FailSafeOracle
from aalpy.SULs import StochasticMealySUL
from aalpy.automata import Onfsm
from aalpy.base import SUL
from aalpy.learning_algs import run_non_det_Lstar, run_Alergia
# from FailSafeOracle import FailSafeOracle
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle
from aalpy.utils import load_automaton_from_file
from aalpy.learning_algs import run_stochastic_Lstar
from FailSafeLearning.StatePrefixEqOracleFailSafe import StatePrefixOracleFailSafe

# def test_counter():
from tester import FailSUL

serial_port = sys.argv[1]
advertiser_address = sys.argv[2]


query_log_file = os.path.join(os.getcwd(), "query_logs/query_log.txt")
if os.path.exists(query_log_file):
    shutil.copy2(query_log_file, os.path.join(os.getcwd(), "query_logs/query_log_temp.txt"))
file = open(query_log_file, "w")
file.close()

query_log_file = os.path.join(os.getcwd(), "query_logs/oracle_log.txt")
if os.path.exists(query_log_file):
    shutil.copy2(query_log_file, os.path.join(os.getcwd(), "query_logs/oracle_log_temp.txt"))
file = open(query_log_file, "w")
file.close()


# model = load_automaton_from_file("correct_model.dot", "onfsm")
model = load_automaton_from_file("failed_transition_model.dot", "smm")
alphabet = model.get_input_alphabet()
# alphabet = ['scan_req', 'connection_req', 'length_req', 'length_rsp', 'feature_rsp', 'feature_req', 'version_req',
#             'mtu_req', 'pairing_req']
# print(alphabet)
# sul = BLESUL(serial_port, advertiser_address)
# sul = FailSUL(model)
sul = StochasticMealySUL(model)

# model = load_automaton_from_file("correct_model.dot", "onfsm")

# eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=1000, reset_prob=0.05, reset_after_cex=True)
eq_oracle = StatePrefixOracleFailSafe(alphabet, sul, walks_per_state=100, walk_len=20)

# counterexample = eq_oracle.find_cex(model)
# if counterexample is not None:
#     print(counterexample)
#     exit()

# eq_oracle = FailSafeOracle(alphabet, sul, num_walks=1000, min_walk_len=4, max_walk_len=10, reset_after_cex=False)
# eq_oracle = RandomWordEqOracle(alphabet, sul)

# learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, debug=True, stochastic="smm")

# eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=10, reset_after_cex=False)

# learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, print_level=2, pruning_threshold=0.2, debug=True, stochastic="smm")
learned_model = run_stochastic_Lstar(alphabet, sul, eq_oracle, min_rounds=10, automaton_type="smm", custom_oracle=True, strategy="device")


# learned_model.visualize()

