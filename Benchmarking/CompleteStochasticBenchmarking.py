import random

from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles.RandomWalkEqOracle import UnseenOutputRandomWalkEqOracle
from aalpy.utils import load_automaton_from_file
from aalpy.utils import smm_to_mdp_conversion, model_check_experiment

path_to_dir = '../DotModels/MDPs/'
files = ['first_grid.dot', 'second_grid.dot',
         'slot_machine.dot', 'shared_coin.dot', 'mqtt.dot', 'tcp.dot']

prop_folder = 'prism_eval_props/'

# TODO Change the path to your PRIMS executable
prism_executable = "/home/mtappler/Programs/prism-4.4-linux64/bin/prism"
prism_executable = "C:/Program Files/prism-4.6/bin/prism.bat"

n_c = 20
n_resample = 1000
min_rounds = 10
max_rounds = 1000

uniform_parameters = False
strategy = ["no_cq", "normal", "chi_square"]
cex = [None, 'bfs', 'random:100:0.15']

for strat in strategy:
    for cex_stat in cex:
        for seed in range(1, 21):
            random.seed(seed)
            benchmark_dir = f"benchmark_complete_no_cq/benchmark_{strat}_{seed}"
            import os

            if not os.path.exists(benchmark_dir):
                os.makedirs(benchmark_dir)
            text_file = open(f"{benchmark_dir}/StochasticExperiments.csv", "w")

            for file in files:
                exp_name = file.split('.')[0]
                if uniform_parameters:
                    if exp_name == 'first_grid':
                        n_c, n_resample = n_c, n_resample
                    elif exp_name == 'second_grid':
                        n_c, n_resample = n_c, n_resample
                    elif exp_name == 'shared_coin':
                        n_c, n_resample = n_c, n_resample
                    elif exp_name == 'slot_machine':
                        n_c, n_resample = n_c, n_resample
                    elif exp_name == 'mqtt':
                        n_c, n_resample = n_c, n_resample
                    elif exp_name == 'tcp':
                        n_c, n_resample = n_c, n_resample
                else:
                    if exp_name == 'first_grid':
                        n_c, n_resample = 20, 1000
                    elif exp_name == 'second_grid':
                        n_c, n_resample = 20, 1500
                    elif exp_name == 'shared_coin':
                        n_c, n_resample = 25, 3000
                    elif exp_name == 'slot_machine':
                        n_c, n_resample = 40, 5000
                    elif exp_name == 'mqtt':
                        n_c, n_resample = 20, 1000
                    elif exp_name == 'tcp':
                        n_c, n_resample = 20, 1000

                original_mdp = load_automaton_from_file(path_to_dir + file, automaton_type='mdp')
                input_alphabet = original_mdp.get_input_alphabet()

                mdp_sul = MdpSUL(original_mdp)

                eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, mdp_sul, num_steps=n_resample * (1 / 0.25),
                                                           reset_after_cex=True, reset_prob=0.25)

                learned_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='mdp',
                                                             n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, strategy=strategy,
                                                             max_rounds=max_rounds, return_data=True, samples_cex_strategy=cex_stat)

                mdp_sul.num_steps = 0
                mdp_sul.num_queries = 0

                learned_smm, data_smm = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='smm',
                                                             n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, strategy=strategy,
                                                             max_rounds=max_rounds, return_data=True, samples_cex_strategy=cex_stat)

                smm_2_mdp = smm_to_mdp_conversion(learned_smm)

                mdp_results, mdp_err = model_check_experiment(prism_executable, exp_name, learned_mdp, prop_folder)
                smm_results, smm_err = model_check_experiment(prism_executable, exp_name, smm_2_mdp, prop_folder)

                properties_string_header = ",".join([f'{key}_val,{key}_err' for key in mdp_results.keys()])

                property_string_mdp = ",".join([f'{str(mdp_results[p])},{str(mdp_err[p])}' for p in mdp_results.keys()])
                property_string_smm = ",".join([f'{str(smm_results[p])},{str(smm_err[p])}' for p in smm_results.keys()])

                text_file.write('Exp_Name, n_c, n_resample, Final Hypothesis Size, Learning time,'
                                'Eq. Query Time, Learning Rounds, #MQ Learning, # Steps Learning,'
                                f'# MQ Eq.Queries, # Steps Eq.Queries , {properties_string_header}\n')

                text_file.write(f'learned_mdp_{exp_name},{n_c},{n_resample}, {data_mdp["automaton_size"]}, '
                                f'{data_mdp["learning_time"]}, {data_mdp["eq_oracle_time"]}, '
                                f'{data_mdp["learning_rounds"]}, {data_mdp["queries_learning"]}, {data_mdp["steps_learning"]},'
                                f'{data_mdp["queries_eq_oracle"]}, {data_mdp["steps_eq_oracle"]},'
                                f'{property_string_mdp}\n')

                text_file.write(f'learned_smm_{exp_name},{n_c},{n_resample}, {data_smm["automaton_size"]}, '
                                f'{data_smm["learning_time"]}, {data_smm["eq_oracle_time"]}, '
                                f'{data_smm["learning_rounds"]}, {data_smm["queries_learning"]}, {data_smm["steps_learning"]},'
                                f'{data_smm["queries_eq_oracle"]}, {data_smm["steps_eq_oracle"]},'
                                f'{property_string_smm}\n')

                text_file.flush()

            text_file.close()
