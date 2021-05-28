import os
import stormpy
import random
from aalpy.automata import Mdp
from aalpy.SULs import MdpSUL
from aalpy.utils import get_faulty_coffee_machine_MDP


class PDS:
    def __init__(self,
                 target,
                 k,
                 input_alphabet=None,
                 p_rand=0,
                 c_change=0.9,
                 s_all=None,
                 formula_str="",
                 sul=None):
        self.target = target
        self.k = k
        self.input_alphabet = input_alphabet if input_alphabet is not None else []
        self.p_rand = p_rand
        self.c_change = c_change
        self.s_all = s_all if s_all is not None else []
        self.formula_str = formula_str
        self.sul = sul

    def execute_pds(self, mdp):
        """
        Args:
            mdp:

        Returns:
        """

        prism_program = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + "/prism_files/coffee_machine_prism.txt"

        if type(mdp) == str:
            # TODO: read and transform prism file
            self.sul = MdpSUL(get_faulty_coffee_machine_MDP())
            prism_program = stormpy.parse_prism_program(path, simplify=False)
        elif type(mdp) == Mdp:
            # TODO: transform MDP to stormpy format
            self.sul = MdpSUL(mdp)
            prism_program = stormpy.parse_prism_program(path, simplify=False)

        self.input_alphabet = self.sul.mdp.get_input_alphabet()
        self.formula_str = self.build_formula(self.formula_str,
                                              [state.output for state in self.sul.mdp.states],
                                              self.target,
                                              self.k)
        if prism_program is not None and self.sul is not None:
            formulas = stormpy.parse_properties(self.formula_str, prism_program)
            hypothesis = stormpy.build_model(prism_program, formulas)

            result = stormpy.model_checking(hypothesis, formulas[0], extract_scheduler=True)
            assert result.has_scheduler
            scheduler = result.scheduler
            assert scheduler.memoryless
            assert scheduler.deterministic

            return self.pds(self.p_rand, self.sul, scheduler, hypothesis, self.k+1, self.k, self.s_all, self.c_change)

    @staticmethod
    def pds(p_rand, sul, scheduler, hypothesis, n_batch, k, s_all, c_change):
        """
        Args:
            p_rand:     possibility to choose inputs randomly
            sul:        system under learning
            scheduler:
            hypothesis: mdp that is to examine
            n_batch:    number of rounds the algorithm samples
            k:
            s_all:      set of all traces sampled so far
            c_change:

        Returns:
            p_rand:     possibility to choose random inputs in the final round
            s_next:     set of traces that were sampled
            s_all:      set of all traces sampled so far
        """

        def coin_flip(p_rand):
            """
            Args:
                p_rand:

            Returns:
            """

            return random.choices([True, False], weights=[p_rand, 1 - p_rand], k=1)[0]

        def rand_sel(input_set):
            """
            Args:
                input_set:

            Returns:
            """

            return random.choices(input_set, weights=[1 / len(input_set)] * len(input_set), k=1)[0]

        def get_corresponding_input(input, input_alphabet):
            """
            Args:
                input:
                input_alphabet:

            Returns:
            """

            if type(input) == str:
                return input_alphabet.index(input)
            elif type(input) == int:
                return input_alphabet[input]

        def transition_function(q_curr, input, input_alphabet):
            """
            Args:
                self:
                q_curr:
                input:
                input_alphabet:

            Returns:
            """

            dist_q = []
            if q_curr is not None:
                for action in hypothesis.states[q_curr].actions[
                        get_corresponding_input(input, input_alphabet)].transitions:
                    dist_q.append(action)
            return dist_q

        def sample(p_rand, sul, scheduler, hypothesis, k):
            """
            Args:
                p_rand:
                sul:
                scheduler:
                hypothesis:
                k:

            Returns:
            """

            trace = [sul.mdp.initial_state.output]
            sul.mdp.reset_to_initial()
            q_curr = hypothesis.initial_states[0]

            while len(trace) - 1 < k or not coin_flip(0.3):
                if coin_flip(p_rand) or q_curr is None or not sul.mdp.get_input_alphabet()[
                        scheduler.get_choice(q_curr).get_deterministic_choice()]:
                    input = rand_sel(sul.mdp.get_input_alphabet())
                else:
                    input = sul.mdp.get_input_alphabet()[scheduler.get_choice(q_curr).get_deterministic_choice()]

                out_sut = sul.step(input)
                trace.append(input)
                trace.append(out_sut)
                dist_q = transition_function(q_curr, input, sul.mdp.get_input_alphabet())

                for entry in dist_q:
                    if out_sut in hypothesis.states[entry.column].labels:
                        q_curr = entry.column
                        break
                else:
                    q_curr = None

            return trace

        s_next = []
        while len(s_next) < n_batch:
            s_next.append(sample(p_rand, sul, scheduler, hypothesis, k))

        p_rand *= c_change
        s_all.append(s_next)
        return p_rand, s_next, s_all

    @staticmethod
    def build_formula(formula_str, states, target, k):
        formula_str += "Pmax=? ["
        for input in states:
            formula_str += f"\"{input}\" | "
        formula_str = formula_str[0:len(formula_str)-2]
        formula_str += f"U \"{target}\" & steps < {k}]"

        return formula_str
