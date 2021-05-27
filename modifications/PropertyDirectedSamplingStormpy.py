import os
import stormpy
import random
from aalpy.automata import Mdp
from aalpy.base import Oracle, SUL


# TODO: create function to add lines to prism file
# TODO: Maybe check for fixing the quickfix in the formula?
# TODO: revisit the implementation of the algorithm and implementing feedback
# TODO: Advance further on the path of heaven and earth checking!


class PDS(Oracle):
    # TODO: Check for n_batch
    def __init__(self, alphabet: list,
                 sul: SUL,
                 target: str,
                 k: int,
                 p_rand: float = 0.0,
                 c_change: float = 0.9,
                 s_all: list = None,
                 formula_str: str = "",
                 n_batch: int = 0,
                 quit_prob: float = 0.3,
                 stop_on_cex=False):
        super().__init__(alphabet, sul)
        self.target = target
        self.k = k
        self.p_rand = p_rand
        self.c_change = c_change
        self.s_all = s_all if s_all is not None else []
        self.formula_str = formula_str
        self.n_batch = n_batch              # TODO: n_batch = BOUND?
        self.quit_prob = quit_prob
        self.stop_on_cex = stop_on_cex

    def find_cex(self, hypothesis):
        prism_program = None

        if type(hypothesis) == str:
            prism_program = self.parse_prism_file(hypothesis)
        elif type(hypothesis) == Mdp:
            prism_program = self.parse_aalpy_mdp(hypothesis)

        self.formula_str = self.build_formula(self.formula_str,
                                              [state.output for state in self.sul.mdp.states],
                                              self.target,
                                              self.k)

        if prism_program is not None:
            formulas = stormpy.parse_properties(self.formula_str, prism_program)
            hypothesis = stormpy.build_model(prism_program, formulas)
            result = stormpy.model_checking(hypothesis, formulas[0], extract_scheduler=True)

            return self.pds(self.p_rand, hypothesis, result.scheduler, self.k+1, self.k, self.s_all, self.c_change)

    def pds(self, p_rand, hypothesis, scheduler, n_batch, k, s_all, c_change):
        s_next = []
        while len(s_next) < n_batch:
            s_next.append(self.sample(p_rand, scheduler, hypothesis, k))

        p_rand *= c_change
        s_all.append(s_next)
        return p_rand, s_next, s_all

    def sample(self, p_rand, scheduler, hypothesis, k):
        # TODO: include stop_on_cex
        trace = [self.sul.mdp.initial_state.output]
        self.sul.mdp.reset_to_initial()
        q_curr = hypothesis.initial_states[0]

        while len(trace) - 1 < k or not self._coin_flip(self.quit_prob):
            if self._coin_flip(p_rand) or q_curr is None or not self.sul.mdp.get_input_alphabet()[
                    scheduler.get_choice(q_curr).get_deterministic_choice()]:
                input = self._rand_sel(self.sul.mdp.get_input_alphabet())
            else:
                input = self.sul.mdp.get_input_alphabet()[scheduler.get_choice(q_curr).get_deterministic_choice()]

            out_sut = self.sul.step(input)
            trace.append(input)
            trace.append(out_sut)
            dist_q = self._transition_function(q_curr, input, self.sul.mdp.get_input_alphabet(), hypothesis)

            for entry in dist_q:
                if out_sut in hypothesis.states[entry.column].labels:
                    q_curr = entry.column
                    break
            else:
                # TODO: return and handle counterexample(trace) here
                q_curr = None

        return trace

    @staticmethod
    def parse_aalpy_mdp(mdp):
        # TODO: How to create stormpy-mdp here? Maybe create prism parsed output and hand it to stormpy?
        # TODO: Question: What happened to the prism dump method?

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + "/coffee_machine_prism.txt"
        prism_program = stormpy.parse_prism_program(path, simplify=False)
        return prism_program

    @staticmethod
    def parse_prism_file(hypothesis):
        prism_program = stormpy.parse_prism_program(hypothesis, simplify=False)
        return prism_program

    @staticmethod
    def _coin_flip(p_rand):
        return random.choices([True, False], weights=[p_rand, 1 - p_rand], k=1)[0]

    @staticmethod
    def _rand_sel(input_set):
        return random.choices(input_set, weights=[1 / len(input_set)] * len(input_set), k=1)[0]

    @staticmethod
    def _get_corresponding_input(input, alphabet):
        if type(input) == str:
            return alphabet.index(input)
        elif type(input) == int:
            return alphabet[input]

    def _transition_function(self, q_curr, input, alphabet, hypothesis):
        dist_q = []
        if q_curr is not None:
            for action in hypothesis.states[q_curr].actions[
                    self._get_corresponding_input(input, alphabet)].transitions:
                dist_q.append(action)
        return dist_q

    @staticmethod
    def build_formula(formula_str, states, target, k):
        # TODO: Find a better solution for this problem
        formula_str += "Pmax=? ["
        for input in states:
            formula_str += f"\"{input}\" | "
        formula_str = formula_str[0:len(formula_str)-2]
        formula_str += f"U \"{target}\" & steps < {k}]"

        return formula_str

    #---------------------------------------------------------------------------

    def evaluate_step(self):
        return

    def get_new_trace(self):
        return

    def evaluate_trace(self, trace):
        return

    def get_trace_bound(self):
        return