import os
import stormpy
import random
from aalpy.SULs import MdpSUL
from aalpy.utils import get_faulty_coffee_machine_MDP

dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "/coffee_machine_prism.txt"
prism_program = stormpy.parse_prism_program(path, simplify=False)
# formula_str = "Pmax=? [F \"coffee\" & steps < 5]"
formula_str = "Pmax=? [\"beep\" | \"initial\" | \"coffee\" U \"coffee\" & steps < 5]"
formulas = stormpy.parse_properties(formula_str, prism_program)
# opt = stormpy.BuilderOptions().set_build_all_labels()
# model = stormpy.build_sparse_model_with_options(prism_program, formulas, options=opt)
model = stormpy.build_model(prism_program, formulas)


def pds(p_rand, sul, s_h, n_batch, phi, s_all, c_change):
    s_next = []

    k = 0
    while len(s_next) < n_batch:
        s_next.append(sample(p_rand, sul, s_h, 5))
        k += 1

    p_rand_next = p_rand * c_change
    s_all.append(s_next)
    return p_rand_next, s_next, s_all


def sample(p_rand, sul, s_h, k):
    trace = [sul.mdp.initial_state.output]
    sul.mdp.reset_to_initial()
    q_curr = model.initial_states[0]

    while len(trace) - 1 < k or not coin_flip(0.3):
        if coin_flip(p_rand) or q_curr is None or not sul.mdp.get_input_alphabet()[s_h.get_choice(q_curr).get_deterministic_choice()]:
            input = rand_sel(sul.mdp.get_input_alphabet())
        else:
            input = sul.mdp.get_input_alphabet()[s_h.get_choice(q_curr).get_deterministic_choice()]

        out_sut = sul.step(input)
        trace.append(input)
        trace.append(out_sut)
        dist_q = transition_function(q_curr, input, sul.mdp.get_input_alphabet())

        for entry in dist_q:
            if out_sut in model.states[entry.column].labels:
                q_curr = entry.column
                break
        else:
            q_curr = None

    return trace


def coin_flip(p_rand):
    return random.choices([True, False], weights=[p_rand, 1-p_rand], k=1)[0]


def rand_sel(input_set):
    return random.choices(input_set, weights=[1 / len(input_set)] * len(input_set), k=1)[0]


def transition_function(q_curr, input, input_alphabet):
    dist_q = []
    if q_curr is not None:
        for action in model.states[q_curr].actions[get_corresponding_input(input, input_alphabet)].transitions:
            dist_q.append(action)
    return dist_q


def get_corresponding_input(input, input_alphabet):
    if type(input) == str:
        return input_alphabet.index(input)
    elif type(input) == int:
        return input_alphabet[input]


if __name__ == "__main__":
    result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)

    assert result.has_scheduler
    scheduler = result.scheduler
    assert scheduler.memoryless
    assert scheduler.deterministic

# -----------------------------------------------------------------------------------------------------------------------

    mdp = get_faulty_coffee_machine_MDP()
    input_alphabet = mdp.get_input_alphabet()
    sul = MdpSUL(mdp)

    p_rand_new, s_new, s_all_new = pds(0, sul, scheduler, 10, 0, [], 0.9)




























    # i in maxRounds
    # p_rand_i = 0

    # p_rand = 0
    # m_h = sul
    # s_h = scheduler
    # n_batch = 10
    # phi = 0
    # S_all = [s_h]
    # c_change = 0.9



# def sample(p_rand, m_h, s_h, k):
#     trace = ["init"]
#     q_curr = m_h.mdp.initial_states[0]
#
#     while len(trace) - 1 < k or not coin_flip(0.1):
#         if coin_flip(p_rand) or not q_curr or not input_alphabet[s_h.get_choice(q_curr).get_deterministic_choice()]:
#             input = rand_sel(input_alphabet)
#         else:
#             input = s_h[q_curr]
#
#         out_sut = m_h.step(input)
#         trace.append(input)
#         trace.append(out_sut)
#         # dist_q_curr = roh_h(q_curr, input)
#     return trace
#
#
# def coin_flip(p_rand):
#     return random.choices([True, False], weights=[p_rand, 1-p_rand], k=1)
#
#
# def rand_sel(input_set):
#     return random.choices(input_set, weights=[1 / len(input_set)] * len(input_set), k=1)[0]
#
#
# import os
# from cgitb import reset
#
# import stormpy
# import random
#
# from aalpy.SULs import MdpSUL
# from aalpy.utils import generate_random_mdp, get_faulty_coffee_machine_MDP
#
# dir_path = os.path.dirname(os.path.realpath(__file__))
# path = dir_path + "/coffee_machine_prism.txt"
# prism_program = stormpy.parse_prism_program(path)
# formula_str = "Pmax=? [F \"coffee\" & steps < 5]"
# input_alphabet = ["butt", "coin"]
#
#
# def pds(p_rand, m_h, s_h, n_batch, phi, s_all, c_change):
#     s_next = []
#
#     k = 0
#     while len(s_next) < n_batch:
#         s_next.append(sample(p_rand, m_h, s_h, k))
#         k += 1
#
#     p_rand_next = p_rand * c_change
#     s_all.append(s_next)
#     return p_rand_next, s_next, s_all
#
#
# def getInputLabels(m_h):
#     label = ""
#     labels = list(m_h.mdp.states[m_h.mdp.initial_states[0]].labels)
#
#     for l in range(len(labels)):
#         label += labels[l]
#         label += ", "
#
#     return label[0: len(label) - 2]
#
#
# if __name__ == "__main__":
#     formulas = stormpy.parse_properties(formula_str, prism_program)
#     model = stormpy.build_model(prism_program, formulas)
#     result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
#
#     assert result.has_scheduler
#     scheduler = result.scheduler
#     assert scheduler.memoryless
#     assert scheduler.deterministic
#
#     # -----------------------------------------------------------------------------------------------------------------------
#
#     sul = MdpSUL(model)
#
#     p_rand_new, s_new, s_all_new = pds(0.5, sul, scheduler, 10, 0, [scheduler], 0.9)
