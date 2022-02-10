from collections import defaultdict

from aalpy.automata import Onfsm, OnfsmState
from aalpy.base import Automaton, SUL
from aalpy.utils.HelperFunctions import all_suffixes


class NonDetObservationTable:

    def __init__(self, alphabet: list, sul: SUL, n_sampling=100):
        """
        Construction of the non-deterministic observation table.

        Args:

            alphabet: input alphabet
            sul: system under learning
            n_sampling: number of samples to be performed for each cell
        """
        assert alphabet is not None and sul is not None

        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S
        self.S_dot_A = []
        self.E = [tuple([a]) for a in alphabet]
        self.T = defaultdict(dict)
        self.n_samples = n_sampling

        self.sul = sul
        empty_word = tuple()

        # Elements of S are in form that is presented in 'Learning Finite State Models of Observable Nondeterministic
        # Systems in a Testing Context'. Each element of S is a (inputs, outputs) tuple, where first element of the
        # tuple are inputs and second element of the tuple are outputs associated with inputs.
        self.S.append((empty_word, empty_word))
        print(f"Initial Set S {self.S}")

    def get_row_to_close(self):
        """
        Get row for that need to be closed.

        Returns:

            row that will be moved to S set and closed
        """
        s_rows = set()
        for s in self.S:
            s_rows.add(self.row_to_hashable(s))

        for t in self.S_dot_A:
            row_t = self.row_to_hashable(t)

            if row_t not in s_rows:
                self.S.append(t)
                print(f"(Input, Output) set S was updated: {self.S} with element {t}")
                # if len(self.S) > 10:
                    # print("WARNING: number of states are greater than they should be for model CC2650")
                self.S_dot_A.remove(t)
                return t

        return None

    def update_extended_S(self, row):
        """
        Helper generator function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        Returns:

            New rows of extended S set.
        """
        s_set = set(self.S)
        extension = []
        for a in self.A:
            # TODO inserted here because of ERROR, probably caused by deleting entries in update_S set
            if a in self.T[row]:
                for t in self.T[row][a]:
                    new_row = (row[0] + a, row[1] + tuple([t]))
                    if new_row not in s_set:
                        extension.append(new_row)

        self.S_dot_A.extend(extension)
        return extension

    def update_obs_table(self, s_set=None, e_set: list = None):
        """
        Perform the membership queries.
        With  the  all-weather  assumption,  each  output  query  is  tried  a  number  of  times  on  the  system,
        and  the  driver  reports  the  set  of  all  possible  outputs.

        Args:

            s_set: Prefixes of S set on which to preform membership queries (Default value = None)
            e_set: Suffixes of E set on which to perform membership queries


        """

        update_S = s_set if s_set else self.S + self.S_dot_A
        update_E = e_set if e_set else self.E

        for s in update_S:
            table_entries = [entry for entry in self.T.keys() if entry[0] == s[0]]
            if len(table_entries) > 1:
                entries_to_delete = []
                sample_counter = 0

                for table_entry in table_entries:
                    breakout_counter = 0
                    while sample_counter < self.n_samples:
                        output = tuple(self.sul.query(s[0]))
                        if output == table_entry[1]:
                            sample_counter += 1

                        breakout_counter += 1
                        if breakout_counter >= self.n_samples * 10:
                            entries_to_delete.append(table_entry)
                            break

                temp_T = self.T
                temp_S = self.S
                temp_S_dot_A = self.S_dot_A
                for entry_to_delete in entries_to_delete:
                    if entry_to_delete in self.S:
                        temp_S.remove(entry_to_delete)
                    if entry_to_delete in self.S_dot_A:
                        temp_S_dot_A.remove(entry_to_delete)
                    del temp_T[entry_to_delete]
                self.T = temp_T
                self.S = temp_S
                self.S_dot_A = temp_S_dot_A

                if s in entries_to_delete:
                    continue


            flag_to_delete = False
            for e in update_E:
                if e not in self.T[s].keys():
                    num_s_e_sampled = 0
                    # print("update_S set")
                    # for entry in update_S:
                    #     print(entry)
                    # print("\nupdate_E set")
                    # print(update_E)
                    # print("\nT[s] set")
                    # for entry in self.T[s]:
                    #     print(entry)
                    # print("")
                    upper_bound_counter = 0
                    while num_s_e_sampled < self.n_samples:
                        # print(s[0] + e)
                        output = tuple(self.sul.query(s[0] + e))
                        # print(output)
                        # Here I basically say... add just the last element of the output if it e is element of alphabet
                        # else add last len(e) outputs
                        o = output[-1] if len(e) == 1 else tuple(output[-len(e):])
                        if "ERROR" not in output[:len(s[1])]:
                            if output[:len(s[1])] == s[1]:
                                self.add_to_T((s[0], output[:len(s[1])]), e, o)
                                num_s_e_sampled += 1
                                # print(f"{num_s_e_sampled}/{self.n_samples}")
                                # print("")
                            else:
                                upper_bound_counter += 1
                        # else:
                        #     print("MISSMATCH:")
                        #     print(f"output: {output[:len(s[1])]}")
                        #     print(f"reference s[1]: {s[1]}")
                        #     print("------------------------------------------")
                        #     print("update_S set")
                        #     for entry in update_S:
                        #         print(entry)
                        #     print("\nupdate_E set")
                        #     print(update_E)
                        #     print("")
                        #     print("------------------------------------------")

                        if upper_bound_counter >= self.n_samples * 10:
                            row = (s[0][:-1], s[1][:-1])
                            # print(f"row to delete from: {row}")
                            a = (s[0][-1],)
                            # print(f"row entry to delete from: {a}")
                            to_delete = s[1][-1]
                            # print(f"entry to delete: {to_delete}")
                            if a in self.T[row]:
                                if to_delete in self.T[row][a]:
                                    # print("commencing delete!")
                                    self.T[row][a].remove(to_delete)

                            temp_T = self.T
                            temp_S = self.S
                            temp_S_dot_A = self.S_dot_A
                            if s in self.S:
                                temp_S.remove(s)
                            if s in self.S_dot_A:
                                temp_S_dot_A.remove(s)
                            del temp_T[s]
                            self.T = temp_T
                            self.S = temp_S
                            self.S_dot_A = temp_S_dot_A

                            temp_update_S = update_S.copy()
                            temp_update_S.remove(s)
                            update_S = temp_update_S
                            flag_to_delete = True
                            break

                    if flag_to_delete:
                        flag_to_delete = False
                        break



    def gen_hypothesis(self) -> Automaton:
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
            for a in self.A:
                # TODO: if inserted here to combat error caused by deleting entries in set S (udpate_S)
                if a in self.T[prefix]:
                    for t in self.T[prefix][a]:
                        key = self.row_to_hashable((prefix[0] + a, prefix[1] + tuple([t])))
                        if key in state_distinguish.keys():
                            state_in_S = state_distinguish[key]
                            assert state_in_S
                            states_dict[prefix].transitions[a[0]].append((t, state_in_S))
                        else:
                            print("what now?")

        assert initial
        automaton = Onfsm(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton

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
        for e in self.E:
            #Uncommented the if here -> because of error occuring!
            if e in self.T[row_prefix].keys():
                row_repr += (frozenset(self.T[row_prefix][e]),)
        return row_repr

    def add_to_T(self, s, e, value):
        """
        Add values to the cell at T[s][e].

        Args:

            s: prefix
            e: element of S
            value: value to be added to the cell


        """
        if e != "ERROR":
            if e not in self.T[s]:
                self.T[s][e] = set()
            self.T[s][e].add(value)

    def cex_processing(self, cex: tuple):
        """
        Suffix processing strategy found in Shahbaz-Groz paper 'Inferring Mealy Machines'.
        It splits the counterexample into prefix and suffix. Prefix is the longest element of the S union S.A that
        matches the beginning of the counterexample. By removing such prefix from counterexample, no consistency check
        is needed.

        Args:

            cex: counterexample (inputs/outputs)

        Returns:
            suffixes to add to the E set

        """
        prefixes = list(self.S + self.S_dot_A)
        prefixes.reverse()
        trimmed_suffix = None

        cex = tuple(cex[0])  # cex[0] are inputs, cex[1] are outputs
        for p in prefixes:
            prefix_inputs = p[0]
            if prefix_inputs == tuple(cex[:len(prefix_inputs)]):
                trimmed_suffix = cex[len(prefix_inputs):]
                break

        if trimmed_suffix:
            suffixes = all_suffixes(trimmed_suffix)
        else:
            suffixes = all_suffixes(cex)
        suffixes.reverse()
        return suffixes
