import os

from BLESUL import BLESUL
from FailSafeLearning.FailSafeCacheSUL import FailSafeCacheSUL

# def get_error_info(ble: BLESUL, cache: FailSafeCacheSUL):
def get_error_info(ble, cache):
    """
    Create error statistics.
    """
    error_info = {
        'non_det_query': cache.non_det_query_counter,
        'non_det_step': cache.non_det_step_counter,
        'connection_error': ble.connection_error_counter
    }
    return error_info


# def print_error_info(ble: BLESUL, cache: FailSafeCacheSUL):
def print_error_info(ble, cache):
    """
    Print error statistics.
    """
    error_info = get_error_info(ble, cache)
  
    print('-----------------------------------')
    print('Connection errors:  {}'.format(error_info['connection_error']))
    print('Non-determinism in learning: {}'.format(error_info['non_det_query']))
    print('Non-determinism in equivalence check: {}'.format(error_info['non_det_step']))
    print('-----------------------------------')


def print_con_info(ble: BLESUL):
    print('-----------------------------------')
    print('Connection errors:  {}'.format(ble.connection_error_counter))
    print('-----------------------------------')


def get_save_location():
    location = os.path.join(os.getcwd(), "saved_states", "CC2640R2")
    return location