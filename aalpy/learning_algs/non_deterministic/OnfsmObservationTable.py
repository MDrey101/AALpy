from collections import defaultdict

from aalpy.automata import Onfsm, OnfsmState
from aalpy.base import Automaton, SUL
from aalpy.utils.HelperFunctions import all_suffixes


class NonDetObservationTable:

    def __init__(self, alphabet: list, sul: SUL, n_sampling=100, trace_tree=False):
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
        self.trace_tree_flag = trace_tree

        self.sul = sul
        empty_word = tuple()

        # Elements of S are in form that is presented in 'Learning Finite State Models of Observable Nondeterministic
        # Systems in a Testing Context'. Each element of S is a (inputs, outputs) tuple, where first element of the
        # tuple are inputs and second element of the tuple are outputs associated with inputs.
        self.S.append((empty_word, empty_word))
        print(f"Initial Set S {self.S}")

        self.num_non_observed_entries = 0

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
                if len(t) < len(self.E) - 1:
                    continue
                self.S.append(t)
                print(f"Set S was updated: {self.S} with element {t}")
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

        # execute the current s[0] and check the table with that output
        # if entries cannot be observed - insert Empty
        # at the last step - delete all empty entries
        # change update_S and loop through it below
        # for s in update_S:
        #     if s == ((),()):
        #         continue
        #
        #     sample_counter = 0
        #     while sample_counter < self.n_samples:
        #         output = tuple(self.sul.query(s[0]))
        #         if "ERROR" not in output:
        #             new_s = (s[0], output)
        #             if new_s != s:
        #                 print(f"Found different entries: \n{s}\nreplaced by\n{new_s}")
        #                 temp_S = update_S.copy()
        #                 # temp_S[update_S.index(s)] = new_s
        #                 update_S = temp_S
        #                 break
        #             if (s[0], output) == s:
        #                 break
        #             sample_counter += 1
    
        
        
        
        
        
        
        
        
        
        
        
        
        # for s in update_S:
        #     table_entries = [entry for entry in self.T.keys() if entry[0] == s[0]]
            
            # execute the current s[0] and check the table with that output
            # if entries cannot be observed - insert Empty
            # at the last step - delete all empty entries
            # change update_S and loop through it below
            
            
            
            # if len(table_entries) > 1:
            #     entries_to_delete = []
            #     sample_counter = 0
            #
            #     for table_entry in table_entries:
            #         breakout_counter = 0
            #         while sample_counter < self.n_samples:
            #             output = tuple(self.sul.query(s[0]))
            #             if output == table_entry[1]:
            #                 sample_counter += 1
            #
            #             breakout_counter += 1
            #             if breakout_counter >= self.n_samples * 10:
            #                 entries_to_delete.append(table_entry)
            #                 break
            #
            #     temp_T = self.T
            #     temp_S = self.S
            #     temp_S_dot_A = self.S_dot_A
            #     for entry_to_delete in entries_to_delete:
            #         if entry_to_delete in self.S:
            #             temp_S.remove(entry_to_delete)
            #         if entry_to_delete in self.S_dot_A:
            #             temp_n(table_entries) > 1:
            #     entries_to_delete = []
            #     sample_counter = 0
            #
            #     for table_entry in table_entries:
            #         breakout_counter = 0
            #         while sample_counter < self.n_samples:
            #             output = tuple(self.sul.query(s[0]))
            #             if output == table_entry[1]:
            #                 sample_counter += 1
            #
            #             breakout_counter += 1
            #             if breakout_counter >= self.n_samples * 10:
            #                 entries_to_delete.append(table_entry)
            #                 break
            #
            #     temp_T = self.T
            #     temp_S = self.S
            #     temp_S_dot_A = self.S_dot_A
            #     for entry_to_delete in entries_to_delete:
            #         if entry_to_delete in self.S:
            #             temp_S.remove(entry_to_delete)
            #         if entry_to_delete in self.S_dot_A:
            #             temp_S_dot_A.remove(entry_to_delete)
            #         del temp_T[entry_to_delete]
            #     self.T = temp_T
            #     self.S = temp_S
            #     self.S_dot_A = temp_S_dot_A
            #
            #     if s in entries_to_delete:
            #         continue

        observed_s_list = []
        for s in update_S:
            observed_s_list.append(s)
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
                    error_counter = 0
                    while num_s_e_sampled < self.n_samples * max(len(s[0]), 1):
                        print(s[0] + e)
                        print(f"query tries: {num_s_e_sampled}/{self.n_samples * 10* max(len(s[0]), 1)}")
                        output = tuple(self.sul.query(s[0] + e))
                        # print(output)
                        # Here I basically say... add just the last element of the output if it e is element of alphabet
                        # else add last len(e) outputs
                        o = output[-1] if len(e) == 1 else tuple(output[-len(e):])
                        if "ERROR" not in output[:len(s[1])]:
                            # if output[:len(s[1])] == s[1]:
                            if output[:len(s[1])] in [entry[1] for entry in self.T.keys()]:
                                if (s[0], output[:len(s[1])]) not in observed_s_list:
                                    observed_s_list.append((s[0], output[:len(s[1])]))
                                self.add_to_T((s[0], output[:len(s[1])]), e, o)
                                num_s_e_sampled += 1
                                print(f"successful samples: {num_s_e_sampled}/{self.n_samples * max(len(s[0]), 1)}\n")
                            else:
                                self.num_non_observed_entries += 1
                                print(f"# Observed output not in table: {self.num_non_observed_entries}, {output}")
                                upper_bound_counter += 1
                        else:
                            error_counter += 1
                            print(f"error counter: {error_counter}/{self.n_samples*20}")
                            if error_counter >= self.n_samples * 20:
                                print("breaking from loop, because of constant ERROR!")
                                break
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

                        if upper_bound_counter >= self.n_samples * 10 * max(len(s[0]), 1):
                            print(f"Writing Epsilon to {s}[e]")
                            self.add_to_T((s[0], output[:len(s[1])]), e, "Epsilon")
                            break

                            # row = (s[0][:-1], s[1][:-1])
                            # # print(f"row to delete from: {row}")
                            # a = (s[0][-1],)
                            # # print(f"row entry to delete from: {a}")
                            # to_delete = s[1][-1]
                            # # print(f"entry to delete: {to_delete}")
                            # if a in self.T[row]:
                            #     if to_delete in self.T[row][a]:
                            #         # print("commencing delete!")
                            #         self.T[row][a].remove(to_delete)
                            #
                            # temp_T = self.T.copy()
                            # temp_S = self.S.copy()
                            # temp_S_dot_A = self.S_dot_A.copy()
                            # if s in self.S:
                            #     temp_S.remove(s)
                            # if s in self.S_dot_A:
                            #     temp_S_dot_A.remove(s)
                            # del temp_T[s]
                            # self.T = temp_T
                            # self.S = temp_S
                            # self.S_dot_A = temp_S_dot_A
                            #
                            # temp_update_S = update_S.copy()
                            # temp_update_S.remove(s)
                            # update_S = temp_update_S
                            # flag_to_delete = True
                            # break

                    if flag_to_delete:
                        flag_to_delete = False
                        break

            print(observed_s_list)
            for observed_output in observed_s_list:
                if observed_output not in self.T:
                    continue

                if {"Epsilon"} in self.T[observed_output].values() or len(self.T[observed_output]) < len(self.E):
                    epsilon_counter = 0
                    epsilon_entry = None
                    table_entry = None
    
                    print("Checking for Epsilon entries")
                    for key, value in self.T[observed_output].items():
                        if epsilon_counter >= 2:
                            print("2 Epsilon entries found - commencing to adapt the Table!")
                            table_entry = self.delete_entry(observed_output, update_S)
                        if value == {"Epsilon"}:
                            epsilon_entry = key
                            epsilon_counter += 1

                    query_repeat_counter = 0
                    if (epsilon_entry is None and table_entry is None) or table_entry is not None:
                        if table_entry is None:
                            table_entry = self.delete_entry(observed_output, update_S)

                        if table_entry is None:
                            continue

                        row = table_entry[0]
                        key = table_entry[1][0]
                        removed_output = table_entry[1][1]

                        while query_repeat_counter < 100:
                            output = tuple(self.sul.query(row + key))
                            o = output[-1] if len(key) == 1 else tuple(output[-len(key):])
                            if "ERROR" not in output[:len(observed_output[1])]:
                                if output[:len(observed_output[1])] == observed_output[1] and o != removed_output:
                                    self.add_to_T((observed_output[0], output[:len(observed_output[1])]), key, o)
                                    self.T[row][key].remove("Epsilon")
                                    break
                                query_repeat_counter += 1
                        if query_repeat_counter >= 100:
                            print(f"Resampling {row, key} was not successfull!")
                            exit(-1)
    
                    else:
                        print(f"Resampling Epsilon entry of {observed_output}[{epsilon_entry}]")
                        while query_repeat_counter < 100:
                            output = tuple(self.sul.query(observed_output[0] + epsilon_entry))
                            o = output[-1] if len(epsilon_entry) == 1 else tuple(output[-len(epsilon_entry):])
                            if "ERROR" not in output[:len(observed_output[1])]:
                                if output[:len(observed_output[1])] == observed_output[1]:
                                    self.add_to_T((observed_output[0], output[:len(observed_output[1])]), epsilon_entry, o)
                                    self.T[observed_output][epsilon_entry].remove("Epsilon")
                                    break
                                query_repeat_counter += 1
                        if query_repeat_counter >= 100:
                            print(f"Resampling {observed_output[0], epsilon_entry} was not successfull!")
                            exit(-1)
                    # else:
                    #     row = table_entry[0]
                    #     key = table_entry[1][0]
                    #     removed_output = table_entry[1][1]
                    #
                    #     while query_repeat_counter < 100:
                    #         output = tuple(self.sul.query(row + key))
                    #         o = output[-1] if len(key) == 1 else tuple(output[-len(key):])
                    #         if "ERROR" not in output[:len(s[1])]:
                    #             if output[:len(s[1])] == s[1] and o != removed_output:
                    #                 self.add_to_T((s[0], output[:len(s[1])]), key, o)
                    #                 self.T[row][key].remove("Epsilon")
                    #                 break
                    #             query_repeat_counter += 1


    def delete_entry(self, s, update_S):
        return_entry = None
        
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
                if self.T[row][a] is set() or self.T[row][a] == ():
                    self.T[row][a].add("Epsilon")
                    return_entry = (row, (a, to_delete))
    
        temp_T = self.T.copy()
        temp_S = self.S.copy()
        temp_S_dot_A = self.S_dot_A.copy()
        if s in self.S:
            temp_S.remove(s)
        if s in self.S_dot_A:
            temp_S_dot_A.remove(s)
        del temp_T[s]
        self.T = temp_T
        self.S = temp_S
        self.S_dot_A = temp_S_dot_A
    
        temp_update_S = update_S.copy()
        if s in temp_update_S:
            temp_update_S.remove(s)
        update_S = temp_update_S
        flag_to_delete = True
        return return_entry


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

        # if self.trace_tree_flag:
        #     for prefix in self.S:
        #         curr_node = self.sul.pta.get_to_node(prefix[0], prefix[1])
        #         for a in self.A:
        #             for t in self.sul.pta.get_single_trace(curr_node, a):
        #                 if self.row_to_hashable((prefix[0] + a, prefix[1] + tuple(t))) in state_distinguish.keys():
        #                     state_in_S = state_distinguish[self.row_to_hashable((prefix[0] + a, prefix[1] + tuple(t)))]
        #                     assert state_in_S
        #                     states_dict[prefix].transitions[a[0]].append((t[0], state_in_S))
        # else:
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

            row_prefix: prefixkey of the row in the observation table

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
