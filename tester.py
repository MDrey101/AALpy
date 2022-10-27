from random import choice, choices

from aalpy.automata import Onfsm
from aalpy.base import SUL
from aalpy.learning_algs import run_non_det_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.SULs import OnfsmSUL
from aalpy.oracles.FailSafeOracle import FailSafeOracle
from aalpy.utils import load_automaton_from_file

model = load_automaton_from_file("fail_safe_model.dot", "onfsm")
alphabet = model.get_input_alphabet()

# TODO: method which takes the pruned tree - queries the tree?


class FailSUL(SUL):
    def __init__(self, mdp: Onfsm):
        super().__init__()
        self.onfsm = mdp

    def pre(self):
        self.onfsm.reset_to_initial()

    def post(self):
        pass

    def step(self, letter):
        if "DANGER" in self.onfsm.outputs_on_input(letter):
            possible_states = self.onfsm.current_state.transitions[letter]
            danger_position = ["DANGER" == state[0] for state in possible_states]
            probability_distributions = [0.1 if d else 0.9 for d in danger_position]
            transition = choices(possible_states, probability_distributions, k=1)[0]
        else:
            transition = choice(self.onfsm.current_state.transitions[letter])
        output = transition[0]
        self.onfsm.current_state = transition[1]
        return output


sul = FailSUL(model)

eq_oracle = FailSafeOracle(alphabet, sul, num_walks=1000, min_walk_len=4, max_walk_len=8)

learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10, custom_oracle=True)

learned_model.visualize()