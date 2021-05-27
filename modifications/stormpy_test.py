import os
import stormpy

dir_path = os.path.dirname(os.path.realpath(__file__))

path = dir_path + "/coffee_machine_prism.txt"
# path = dir_path + "/test.txt"
prism_program = stormpy.parse_prism_program(path)

model = stormpy.build_model(prism_program)
# model = stormpy.build_sparse_model(prism_program)
print(f"Number of states: {model.nr_states}")
print(f"Number of transitions: {model.nr_transitions}")
print(f"Labels: {model.labeling.get_labels()}\n")

# formula_str = "Pmax=? [F \"coffee\" & steps < 5]"
formula_str = "Pmax=? [F \"beep\" & steps > 5]"
# formula_str = "Pmin=? [F \"coffee\"]"
formulas = stormpy.parse_properties(formula_str, prism_program)
model = stormpy.build_model(prism_program, formulas)

result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)

assert result.has_scheduler
scheduler = result.scheduler
assert scheduler.memoryless
assert scheduler.deterministic
print(scheduler)

for state in model.states:
    choice = scheduler.get_choice(state)
    action = choice.get_deterministic_choice()
    print(f"In state {state} choose action {action}")

print("\n___________________________________________________________________\n")
for state in model.states:
    for action in state.actions:
        for transition in action.transitions:
            print("From state {}, with probability {}, go to state {}".format(state, transition.value(), transition.column))


# print("\n___________________________________________________________________\n")
# model = model.apply_scheduler(scheduler)
# print(f"Number of states: {model.nr_states}")
# print(f"Number of transitions: {model.nr_transitions}")
# print(f"Labels: {model.labeling.get_labels()}\n")
#
# result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)
#
# assert result.has_scheduler
# scheduler = result.scheduler
# assert scheduler.memoryless
# assert scheduler.deterministic
# print(scheduler)
#
# for state in model.states:
#     choice = scheduler.get_choice(state)
#     action = choice.get_deterministic_choice()
#     print(f"In state {state} choose action {action}")
#
# print("\n___________________________________________________________________\n")
# for state in model.states:
#     for action in state.actions:
#         for transition in action.transitions:
#             print("From state {}, with probability {}, go to state {}".format(state, transition.value(), transition.column))
