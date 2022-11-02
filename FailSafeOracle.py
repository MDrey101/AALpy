from random import randint, choice

from aalpy.automata import Onfsm, Mdp, StochasticMealyMachine
from aalpy.base import Oracle, SUL

automaton_dict = {Onfsm: 'onfsm', Mdp: 'mdp', StochasticMealyMachine: 'smm'}


class FailSafeOracle(Oracle):
    """
    Equivalence oracle where queries are of random length in a predefined range, and cex from a unsafe set are ignored.
    """

    def __init__(self, alphabet: list, sul: SUL, num_walks=100, min_walk_len=10, max_walk_len=100,
                 reset_after_cex=True):
        """
        Args:
            alphabet: input alphabet

            sul: system under learning

            num_walks: number of walks to perform during search for cex

            min_walk_len: minimum length of each walk

            max_walk_len: maximum length of each walk

            reset_after_cex: if True, num_walks will be preformed after every counter example, else the total number
                or walks will equal to num_walks
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.reset_after_cex = reset_after_cex
        self.num_walks_done = 0
        self.automata_type = None

        # set of counterexamples that are deemed as unsafe
        self.unsafe_counterexamples = set()

    def find_cex(self, hypothesis):

        while self.num_walks_done < self.num_walks:
            # print(self.num_walks_done)
            inputs = []
            outputs = []
            self.reset_hyp_and_sul(hypothesis)
            self.num_walks_done += 1

            num_steps = randint(self.min_walk_len, self.max_walk_len)

            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])

                out_hyp = hypothesis.step_to(inputs[-1], out_sul)
                outputs.append(out_sul)

                self.num_steps += 1

                if out_hyp is None:
                    if self.is_cex_dangerous(inputs, outputs):
                        break

                    if self.reset_after_cex:
                        self.num_walks_done = 0
                        self.sul.post()

                    return inputs, outputs

        return None

    def is_cex_dangerous(self, inputs, outputs):
        for index in range(1, len(inputs) + 1):
            i = tuple(inputs[:index])
            o = tuple(outputs[:index])
            if (i, o) in self.unsafe_counterexamples:
                return True
        return False

    def reset_counter(self):
        if self.reset_after_cex:
            self.num_walks_done = 0
