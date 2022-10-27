from collections import Counter

from aalpy.automata import Onfsm, OnfsmState
from aalpy.base import Automaton
from aalpy.learning_algs.non_deterministic.TraceTree import SULWrapper
from aalpy.utils.HelperFunctions import print_observation_table, all_suffixes


class NonDetObservationTable:

    def __init__(self, alphabet: list, sul: SULWrapper, n_sampling):
        """
        Construction of the non-deterministic observation table.

        Args:

            alphabet: input alphabet
            sul: system under learning
            n_sampling: number of samples to be performed for each cell
        """
        assert alphabet is not None and sul is not None

        self.alphabet = alphabet
        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S

        self.E = [tuple([a]) for a in alphabet]

        self.n_samples = n_sampling
        self.closing_counter = 0

        self.sul = sul

        self.sampling_counter = Counter()

        empty_word = tuple()

        # Elements of S are in form that is presented in 'Learning Finite State Models of Observable Nondeterministic
        # Systems in a Testing Context'. Each element of S is a (inputs, outputs) tuple, where first element of the
        # tuple are inputs and second element of the tuple are outputs associated with inputs.
        self.S.append((empty_word, empty_word))

        self.pruned_nodes = set()

    def get_row_to_close(self):
        """
        Get row for that need to be closed.

        Returns:

            row that will be moved to S set and closed
        """

        # pruned = self.sul.pta.prune()
        # self.pruned_nodes.update(pruned)

        s_rows = set()
        update_S_dot_A = self.get_extended_S()

        for s in self.S.copy():
            s_rows.add(self.row_to_hashable(s))

        for t in update_S_dot_A:
            row_t = self.row_to_hashable(t)
            if row_t not in s_rows:
                self.closing_counter += 1
                self.S.append(t)
                return t

        self.closing_counter = 0
        return None

    def get_extended_S(self, row_prefix=None):
        """
        Helper generator function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        Returns:

            extended S set.
        """

        rows = self.S if row_prefix is None else [row_prefix]

        S_dot_A = []
        for row in rows:
            curr_node = self.sul.pta.get_to_node(row[0], row[1])
            for a in self.A:
                trace = self.sul.pta.get_all_traces(curr_node, a)

                for t in trace:
                    new_row = (row[0] + a, row[1] + (t[-1],))
                    if new_row not in self.S:
                        S_dot_A.append(new_row)
        return S_dot_A

    def gen_hypothesis(self):
        """
        Generate automaton based on the values found in the observation table.

        Returns:

            Current hypothesis

        """

        state_distinguish = dict()
        states_dict = dict()
        initial = None

        stateCounter = 0
        for prefix in self.S:
            state_id = f's{stateCounter}'
            states_dict[prefix] = OnfsmState(state_id)

            states_dict[prefix].prefix = prefix
            state_distinguish[self.row_to_hashable(prefix)] = states_dict[prefix]

            if prefix == self.S[0]:
                initial = states_dict[prefix]
            stateCounter += 1

        for prefix in self.S:
            curr_node = self.sul.pta.get_to_node(prefix[0], prefix[1])
            for a in self.A:
                trace = self.sul.pta.get_all_traces(curr_node, a)
                for t in trace:
                    reached_row = (prefix[0] + a, prefix[1] + (t[-1],))
                    if self.row_to_hashable(reached_row) not in state_distinguish.keys():
                        print('HOLE IN THE OBSERVATION TABLE')
                        assert False
                    state_in_S = state_distinguish[self.row_to_hashable(reached_row)]
                    assert state_in_S  # shouldn't be necessary because of the if condition
                    states_dict[prefix].transitions[a[0]].append((t[-1], state_in_S))

        assert initial
        automaton = Onfsm(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton

    def query_holes(self, s=None, e=None):
        s_set = [s] if s is not None else self.S + self.get_extended_S()
        e_set = [e] if e is not None else self.E

        for s in s_set:
            for e in e_set:
                while self.sul.pta.get_s_e_sampling_frequency(s, e) < self.n_samples:
                    self.sul.query(s[0] + e)

    def row_to_hashable(self, row_prefix):
        """
        Creates the hashable representation of the row. Frozenset is used as the order of element in each cell does not
        matter

        Args:

            row_prefix: prefix of the row in the observation table

        Returns:

            hashable representation of the row

        """
        row_repr = tuple()
        curr_node = self.sul.pta.get_to_node(row_prefix[0], row_prefix[1])

        for e in self.E:
            cell = self.sul.pta.get_all_traces(curr_node, e)
            while cell is None:
                self.query_holes(row_prefix, e)
                cell = self.sul.pta.get_all_traces(curr_node, e)

            row_repr += (frozenset(cell),)

        return row_repr

    def is_in_cache(self, s):
        curr_node = self.sul.pta.get_to_node(s[0], s[1])
        return curr_node is not None

    def clean_obs_table(self):
        """
        Moves duplicates from S to S_dot_A. The entries in S_dot_A which are based on the moved row get deleted.
        The table will be smaller and more efficient.

        """

        tmp_S = self.S.copy()
        tmp_both_S = self.S + self.get_extended_S()
        hashed_rows_from_s = set()

        tmp_S.sort(key=lambda t: len(t[0]))

        for s in tmp_S:
            hashed_s_row = self.row_to_hashable(s)
            if hashed_s_row in hashed_rows_from_s:
                if s in self.S:
                    self.S.remove(s)
                size = len(s[0])
                for row_prefix in tmp_both_S:
                    s_both_row = (row_prefix[0][:size], row_prefix[1][:size])
                    if s != row_prefix and s == s_both_row:
                        if row_prefix in self.S:
                            self.S.remove(row_prefix)
            else:
                hashed_rows_from_s.add(hashed_s_row)

    def reconstruct_obs_table(self):

        self.E = [tuple([a]) for a in self.alphabet]

        hypothesis = None

        while True:
            self.S = list()
            self.S.append((tuple(), tuple()))
            self.query_holes()

            row_to_close = self.get_row_to_close()
            i = 0
            while row_to_close is not None:
                i += 1
                self.query_holes()
                row_to_close = self.get_row_to_close()
                self.clean_obs_table()

            hypothesis = self.gen_hypothesis()

            cex = self.sul.pta.find_cex_in_cache(hypothesis)
            if cex is None:
                cex = None # EQ ORACLE

            if cex is None:
                break
            else:
                cex_suffixes = all_suffixes(cex[0])
                for suffix in cex_suffixes:
                    if suffix not in self.E:
                        self.E.append(suffix)
                        break

        return hypothesis
