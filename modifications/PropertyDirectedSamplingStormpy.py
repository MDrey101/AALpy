import os
import StringBuilder
import numpy as numpy
import stormpy
import random
from collections import defaultdict

from numpy.lib import math

from aalpy.automata import Mdp
from aalpy.base import Oracle, SUL
from aalpy.oracles.RandomWalkEqOracle import UnseenOutputRandomWalkEqOracle
from aalpy.utils.ModelChecking import _target_string, _sanitize_for_prism


class PDS(Oracle):
    def __init__(self, alphabet: list,
                 sul: SUL,
                 target: str,
                 k: int,
                 p_rand: float = 0.0,
                 c_change: float = 0.9,
                 n_batch: int = 10,
                 quit_prob: float = 0.3,
                 stop_on_cex=False,
                 epsilon=0.01,
                 delta=0.95):
        super().__init__(alphabet, sul)
        self.target = target
        self.k = k
        self.p_rand = p_rand
        self.c_change = c_change
        self.s_all = []
        self.formula_str = ""
        self.n_batch = n_batch
        self.quit_prob = quit_prob
        self.stop_on_cex = stop_on_cex
        self.epsilon = epsilon
        self.delta = delta
        self.accuracy = 0.0
        self.accuracy_list = []
        self.counterexample_list = []

    def find_cex(self, hypothesis):
        found_target = False
        for state in hypothesis.states:
            if self.target in state.output.split("__"):
                found_target = True

        if not found_target:
            new_oracle = UnseenOutputRandomWalkEqOracle(self.alphabet, self.sul, self.n_batch / self.quit_prob,
                                                        reset_prob=self.quit_prob)
            cex = new_oracle.find_cex(hypothesis)
            self.num_queries += new_oracle.num_queries
            self.num_steps += new_oracle.num_steps
            return cex

        stormpy_hypothesis, scheduler = self._setup_pds(hypothesis)
        if stormpy_hypothesis is not None and scheduler is not None:
            p_rand, s_new, s_all = self.pds(self.p_rand, stormpy_hypothesis, scheduler, self.n_batch, self.k,
                                            self.s_all, self.c_change, self.sul)

            self.num_steps += sum([(len(entry[0]) - 1) // 2 for entry in s_new])
            self.num_queries += len(s_new)
            cex = [trace[0] for trace in s_new if trace[1]]
            cex = cex[0] if len(cex) != 0 else None
            return cex
        else:
            print("Converting hypothesis to stormpy went wrong!")

    def _setup_pds(self, hypothesis):
        prism_program = None
        found_target = False
        for state in hypothesis.states:
            if self.target in state.output.split("__"):
                found_target = True

        if found_target:
            if type(hypothesis) == str:
                prism_program = self.parse_prism_file(hypothesis)
            elif type(hypothesis) == Mdp:
                prism_program = self.parse_aalpy_mdp(hypothesis, "mdp_test", self.k)

            self.formula_str = self.build_formula([state.output for state in hypothesis.states],
                                                  self.target,
                                                  self.k)

            if prism_program is not None:
                formulas = stormpy.parse_properties(self.formula_str, prism_program)
                stormpy_hypothesis = stormpy.build_model(prism_program, formulas)
                result = stormpy.model_checking(stormpy_hypothesis, formulas[0], extract_scheduler=True)
                return stormpy_hypothesis, result.scheduler
        return None, None

    def execute_pds(self, hypothesis, evaluation=False):
        stormpy_hypothesis, scheduler = self._setup_pds(hypothesis=hypothesis)

        if stormpy_hypothesis is not None and scheduler is not None:
            p_rand_new, s_new, s_all_new = self.pds(p_rand=self.p_rand,
                                                    hypothesis=stormpy_hypothesis,
                                                    scheduler=scheduler,
                                                    n_batch=self.n_batch,
                                                    k=self.k,
                                                    s_all=self.s_all,
                                                    c_change=self.c_change,
                                                    sul=self.sul)

            if evaluation:
                saved_prand = self.p_rand
                self.p_rand = 0
                self.accuracy_list.append(self.evaluate_scheduler(scheduler, stormpy_hypothesis, self.sul))
                self.p_rand = saved_prand

            return p_rand_new, s_new, s_all_new
        else:
            print("Converting hypothesis to stormpy went wrong!")

    def pds(self, p_rand, hypothesis, scheduler, n_batch, k, s_all, c_change, sul):
        s_next = []
        while len(s_next) < n_batch:
            trace, counterexample = self.sample(p_rand, scheduler, hypothesis, k, sul)
            s_next.append((trace, counterexample))
            if self.stop_on_cex and counterexample:
                self.counterexample_list.append(trace)
                break

        p_rand *= c_change
        s_all.append(s_next)
        return p_rand, s_next, s_all

    def sample(self, p_rand, scheduler, hypothesis, k, sul):
        sul.post()
        trace = [sul.pre()]
        q_curr = hypothesis.initial_states[0]
        found_counterexample = False

        while ((len(trace) - 1) / 2) - 1 < k or not self._coin_flip(self.quit_prob):
            if self._coin_flip(p_rand) or q_curr is None or not self.alphabet[
                    scheduler.get_choice(q_curr).get_deterministic_choice()]:
                input = self._rand_sel(self.alphabet)
            else:
                input = self.alphabet[scheduler.get_choice(q_curr).get_deterministic_choice()]

            out_sut = sul.step(input)
            trace.append(input)
            trace.append(out_sut)
            dist_q = self._transition_function(q_curr, input, self.alphabet, hypothesis)

            for entry in dist_q:
                if out_sut in hypothesis.states[entry.column].labels:
                    q_curr = entry.column
                    break
            else:
                found_counterexample = True
                q_curr = None

        return trace, found_counterexample

    @staticmethod
    def parse_aalpy_mdp(mdp, name, k):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + f"/prism_files/{name}.txt"

        prism_program_string = PDS.create_prism_program(mdp, "pds_" + name, k)
        try:
            with open(path, "w+") as outfile:
                outfile.write(prism_program_string)
        except Exception:
            return None

        return stormpy.parse_prism_program(path, simplify=False)

    @staticmethod
    def create_prism_program(mdp, name, k):
        builder = StringBuilder.StringBuilder()

        builder.append("mdp")
        builder.append(os.linesep * 2)
        builder.append(f"const int BOUND = {k + 1};")
        builder.append(os.linesep * 2)
        builder.append(f"module {name}")
        builder.append(os.linesep)

        nr_states = len(mdp.states)
        orig_id_to_int_id = dict()
        for index, state in enumerate(mdp.states):
            orig_id_to_int_id[state.state_id] = index
        builder.append("\tloc : [0..{}] init {};".format(nr_states, orig_id_to_int_id[mdp.initial_state.state_id]))
        builder.append(os.linesep)

        # print transitions
        for source in mdp.states:
            source_id = orig_id_to_int_id[source.state_id]
            for inp in source.transitions.keys():
                if source.transitions[inp]:
                    target_strings = \
                        map(lambda target: _target_string(target, orig_id_to_int_id), source.transitions[inp])
                    target_joined = " + ".join(target_strings)
                    builder.append(f"\t[{_sanitize_for_prism(inp)}] loc={source_id} -> {target_joined};")
                    builder.append(os.linesep)
        builder.append("endmodule")
        builder.append(os.linesep * 2)

        builder.append("module StepCounter")
        builder.append(os.linesep)
        builder.append(f"\tsteps: [0..{k + 1}] init 0;")
        builder.append(os.linesep)
        transition_checklist = []
        for source in mdp.states:
            for inp in source.transitions.keys():
                if source.transitions[inp]:
                    if inp not in transition_checklist:
                        transition_checklist.append(inp)
                        builder.append(f"\t[{_sanitize_for_prism(inp)}] true -> 1: (steps'=min(BOUND,steps + 1));")
                        builder.append(os.linesep)
        builder.append("endmodule")
        builder.append(os.linesep * 2)

        # labelling function
        output_to_state_id = defaultdict(list)
        for s in mdp.states:
            joined_output = s.output
            outputs = joined_output.split("__")
            for o in outputs:
                if o:
                    output_to_state_id[o].append(orig_id_to_int_id[s.state_id])

        for output, states in output_to_state_id.items():
            state_propositions = map(lambda s_id: "loc={}".format(s_id), states)
            state_disjunction = "|".join(state_propositions)
            output_string = _sanitize_for_prism(output)
            builder.append(f"label \"{output_string}\" = {state_disjunction};")
            builder.append(os.linesep * 2)

        return builder.to_string()

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

    @staticmethod
    def _transition_function(q_curr, input, alphabet, hypothesis):
        dist_q = []
        if q_curr is not None:
            for action in hypothesis.states[q_curr].actions[
                    PDS._get_corresponding_input(input, alphabet)].transitions:
                dist_q.append(action)
        return dist_q

    @staticmethod
    def build_formula(states, target, k):
        # TODO: Find a better solution for this problem
        formula_str = "Pmax=? ["
        states_checklist = []
        for output in states:
            for out in output.split("__"):
                if out not in states_checklist:
                    states_checklist.append(out)
                    formula_str += f"\"{out}\" | "
        formula_str = formula_str[0:len(formula_str) - 2]
        formula_str += f"U \"{target}\" & steps < {k}]"

        return formula_str

    # ---------------------------------------------------------------------------

    def evaluate_scheduler(self, scheduler, hypothesis, sul):
        sampled_traces = self.get_samples(scheduler, hypothesis, sul)
        num_satisfied_traces = 0
        for trace in sampled_traces:
            for t1 in trace[0][0:2 * self.k:2]:
                if self.target in t1.split("__"):
                    num_satisfied_traces += 1
                    break

        self.accuracy = float(num_satisfied_traces) / float(len(sampled_traces))

        print(self.accuracy)

        return self.accuracy

    def get_samples(self, scheduler, hypothesis, sul):
        sample_traces = []
        for trace_count in range(PDS.get_chernoff_bound(epsilon=self.epsilon, delta=self.delta)):
            sample_traces.append(self.sample(0.0, scheduler, hypothesis, self.k, sul))

        return sample_traces

    @staticmethod
    def get_chernoff_bound(epsilon, delta):
        return math.ceil((numpy.log(2) - numpy.log(delta)) / (2 * epsilon ** 2))
