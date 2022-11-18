import atexit
import os
import sys
import pickle
from BLESUL import BLESUL
from FailSafeOracle import FailSafeOracle
from aalpy.oracles import RandomWordEqOracle
from aalpy.learning_algs import run_non_det_Lstar
from aalpy.learning_algs import run_stochastic_Lstar
from ErrorSULWrapper import ErrorSULWrapper
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper

args_len = len(sys.argv) - 1

if args_len < 2:
    sys.exit(
        "Too few arguments provided.\nUsage: python3 ble_learning.py 'serial_port' 'advertiser_address', ['pcap- & model-filename']")

serial_port = sys.argv[1]
advertiser_address = sys.argv[2]

if args_len == 3:
    pcap_filename = sys.argv[3]
else:
    pcap_filename = 'learning_data'


# def exit_handler():
#     print()
    # with open('trace_tree.pickle', 'wb') as handle:
    #     pickle.dump(sul.cache, handle, protocol=pickle.HIGHEST_PROTOCOL)


ble_sul = BLESUL(serial_port, advertiser_address)
# sul = ble_sul

sul = ErrorSULWrapper(ble_sul)

# sul = NonDeterministicSULWrapper(ble_sul, 0.2)



# if os.path.exists('trace_tree_.pickle'):
#     with open('trace_tree_.pickle', 'rb') as handle:
#         print('Cache initialized')
#         cache = pickle.load(handle)
#         sul.cache = cache

alphabet = ['scan_req', 'connection_req', 'length_req', 'length_rsp', 'feature_rsp', 'feature_req', 'version_req',
            'mtu_req', 'pairing_req']
# alphabet = ['scan_req', 'connection_req', 'length_req']

# no pairing
# alphabet = ['scan_req', 'connection_req', 'length_req', 'length_rsp',  'feature_rsp', 'feature_req', 'version_req', 'mtu_req']

# no length
# alphabet = ['scan_req', 'connection_req', 'length_rsp',  'feature_rsp', 'feature_req', 'version_req', 'mtu_req', 'pairing_req']

# no feature
# alphabet = ['scan_req', 'connection_req', 'length_req', 'length_rsp',  'feature_rsp', 'version_req', 'mtu_req', 'pairing_req']


# eq_oracle = FailSafeOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=10, reset_after_cex=False)
eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=10, reset_after_cex=False)

# atexit.register(exit_handler)

# learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, print_level=2, pruning_threshold=0.2, debug=True,)
learned_model = run_stochastic_Lstar(alphabet, sul, eq_oracle, min_rounds=10, automaton_type="smm", print_level=3)

learned_model.visualize()
