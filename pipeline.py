import os
import subprocess
import concurrent.futures
from eval import evaluate
from eval_analysis import plot_punitiveness_bar_chart, plot_exploitability_bar_chart, plot_punitiveness_vs_exploitability

def run_evals_analysis(args):
    cmd = ['python', 'punitiveness-eval-analysis.py'] + args
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def parallel_processing(func, args_list: list[list]):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, *args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

def main():
    """
    All results, from games, evaluations, and evaluations analysis, are saved in an experiment's subdirectory in the same ./results directory by default
    run_games() - Run the games - comment out if not needed
    
    evaluate() - Run the evaluations - comment out if not needed
    Arguments:
    "-e", "--eval_type", type=str, default="programmatic", help="Evaluation type"
    "-p", "--personas_file", type=str, default="./prompts/personas.json", help="Path to the personas file"
    "-c", "--config_file", type=str, required=True, help="Path to the configuration file"
    "-r", "--results_files", nargs="+", required=True, help="List of paths to the results files"
    "-ref", "--reference_results_files", nargs="+", required=True, help="List of paths to the reference results files"
    "-f", "--focal_agent_name", type=str, required=True, help="Name of the focal agent")
    "-o", "--output_fname", type=str, required=True, help="Base name for output files"
    """
    base_dir = 'results'
    personas_file = 'prompts/personas.json'

    params_tft_neutral = [
        ('gpt-3.5', 'BoS', 'v1', 'TFT', 'baseline'),
        ('gpt-3.5', 'BoS', 'v2', 'TFT', 'baseline'),
        ('gpt-3.5', 'BoS', 'v3', 'TFT', 'baseline'),
        ('gpt-3.5', 'CKN', 'v1', 'TFT', 'baseline'),
        ('gpt-3.5', 'CKN', 'v2', 'TFT', 'baseline'),
        ('gpt-3.5', 'CKN', 'v3', 'TFT', 'baseline'),
        ('gpt-3.5', 'PD', 'v1', 'TFT', 'baseline'),
        ('gpt-3.5', 'PD', 'v2', 'TFT', 'baseline'),
        ('gpt-3.5', 'PD', 'v3', 'TFT', 'baseline'),
        ('claude', 'BoS', 'v1', 'TFT', 'baseline'),
        ('claude', 'BoS', 'v2', 'TFT', 'baseline'),
        ('claude', 'BoS', 'v3', 'TFT', 'baseline'),
        ('claude', 'CKN', 'v1', 'TFT', 'baseline'),
        ('claude', 'CKN', 'v2', 'TFT', 'baseline'),
        ('claude', 'CKN', 'v3', 'TFT', 'baseline'),
        ('claude', 'PD', 'v1', 'TFT', 'baseline'),
        ('claude', 'PD', 'v2', 'TFT', 'baseline'),
        ('claude', 'PD', 'v3', 'TFT', 'baseline'),
        ('mistral', 'BoS', 'v1', 'TFT', 'baseline'),
        ('mistral', 'BoS', 'v2', 'TFT', 'baseline'),
        ('mistral', 'BoS', 'v3', 'TFT', 'baseline'),
        ('mistral', 'CKN', 'v1', 'TFT', 'baseline'),
        ('mistral', 'CKN', 'v2', 'TFT', 'baseline'),
        ('mistral', 'CKN', 'v3', 'TFT', 'baseline'),
        ('mistral', 'PD', 'v1', 'TFT', 'baseline'),
        ('mistral', 'PD', 'v2', 'TFT', 'baseline'),
        ('mistral', 'PD', 'v3', 'TFT', 'baseline'),
        ('gemini', 'BoS', 'v1', 'TFT', 'baseline'),
        ('gemini', 'BoS', 'v2', 'TFT', 'baseline'),
        ('gemini', 'BoS', 'v3', 'TFT', 'baseline'),
        ('gemini', 'CKN', 'v1', 'TFT', 'baseline'),
        ('gemini', 'CKN', 'v2', 'TFT', 'baseline'),
        ('gemini', 'CKN', 'v3', 'TFT', 'baseline'),
        ('gemini', 'PD', 'v1', 'TFT', 'baseline'),
        ('gemini', 'PD', 'v2', 'TFT', 'baseline'),
        ('gemini', 'PD', 'v3', 'TFT', 'baseline'),
    ]

    params_tft_exploiter = [
        ('gpt-3.5', 'BoS', 'v1', 'TFT', 'exploiter'),
        ('gpt-3.5', 'BoS', 'v2', 'TFT', 'exploiter'),
        ('gpt-3.5', 'BoS', 'v3', 'TFT', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v1', 'TFT', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v2', 'TFT', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v3', 'TFT', 'exploiter'),
        ('gpt-3.5', 'PD', 'v1', 'TFT', 'exploiter'),
        ('gpt-3.5', 'PD', 'v2', 'TFT', 'exploiter'),
        ('gpt-3.5', 'PD', 'v3', 'TFT', 'exploiter'),
        ('claude', 'BoS', 'v1', 'TFT', 'exploiter'),
        ('claude', 'BoS', 'v2', 'TFT', 'exploiter'),
        ('claude', 'BoS', 'v3', 'TFT', 'exploiter'),
        ('claude', 'CKN', 'v1', 'TFT', 'exploiter'),
        ('claude', 'CKN', 'v2', 'TFT', 'exploiter'),
        ('claude', 'CKN', 'v3', 'TFT', 'exploiter'),
        ('claude', 'PD', 'v1', 'TFT', 'exploiter'),
        ('claude', 'PD', 'v2', 'TFT', 'exploiter'),
        ('claude', 'PD', 'v3', 'TFT', 'exploiter'),
        ('mistral', 'BoS', 'v1', 'TFT', 'exploiter'),
        ('mistral', 'BoS', 'v2', 'TFT', 'exploiter'),
        ('mistral', 'BoS', 'v3', 'TFT', 'exploiter'),
        ('mistral', 'CKN', 'v1', 'TFT', 'exploiter'),
        ('mistral', 'CKN', 'v2', 'TFT', 'exploiter'),
        ('mistral', 'CKN', 'v3', 'TFT', 'exploiter'),
        ('mistral', 'PD', 'v1', 'TFT', 'exploiter'),
        ('mistral', 'PD', 'v2', 'TFT', 'exploiter'),
        ('mistral', 'PD', 'v3', 'TFT', 'exploiter'),
        ('gemini', 'BoS', 'v1', 'TFT', 'exploiter'),
        ('gemini', 'BoS', 'v2', 'TFT', 'exploiter'),
        ('gemini', 'BoS', 'v3', 'TFT', 'exploiter'),
        ('gemini', 'CKN', 'v1', 'TFT', 'exploiter'),
        ('gemini', 'CKN', 'v2', 'TFT', 'exploiter'),
        ('gemini', 'CKN', 'v3', 'TFT', 'exploiter'),
        ('gemini', 'PD', 'v1', 'TFT', 'exploiter'),
        ('gemini', 'PD', 'v2', 'TFT', 'exploiter'),
        ('gemini', 'PD', 'v3', 'TFT', 'exploiter'),
    ]

    params_neutral_neutral = [
        ('gpt-3.5', 'BoS', 'v1', 'baseline', 'baseline'),
        ('gpt-3.5', 'BoS', 'v2', 'baseline', 'baseline'),
        ('gpt-3.5', 'BoS', 'v3', 'baseline', 'baseline'),
        ('gpt-3.5', 'CKN', 'v1', 'baseline', 'baseline'),
        ('gpt-3.5', 'CKN', 'v2', 'baseline', 'baseline'),
        ('gpt-3.5', 'CKN', 'v3', 'baseline', 'baseline'),
        ('gpt-3.5', 'PD', 'v1', 'baseline', 'baseline'),
        ('gpt-3.5', 'PD', 'v2', 'baseline', 'baseline'),
        ('gpt-3.5', 'PD', 'v3', 'baseline', 'baseline'),
        ('claude', 'BoS', 'v1', 'baseline', 'baseline'),
        ('claude', 'BoS', 'v2', 'baseline', 'baseline'),
        ('claude', 'BoS', 'v3', 'baseline', 'baseline'),
        ('claude', 'CKN', 'v1', 'baseline', 'baseline'),
        ('claude', 'CKN', 'v2', 'baseline', 'baseline'),
        ('claude', 'CKN', 'v3', 'baseline', 'baseline'),
        ('claude', 'PD', 'v1', 'baseline', 'baseline'),
        ('claude', 'PD', 'v2', 'baseline', 'baseline'),
        ('claude', 'PD', 'v3', 'baseline', 'baseline'),
        ('mistral', 'BoS', 'v1', 'baseline', 'baseline'),
        ('mistral', 'BoS', 'v2', 'baseline', 'baseline'),
        ('mistral', 'BoS', 'v3', 'baseline', 'baseline'),
        ('mistral', 'CKN', 'v1', 'baseline', 'baseline'),
        ('mistral', 'CKN', 'v2', 'baseline', 'baseline'),
        ('mistral', 'CKN', 'v3', 'baseline', 'baseline'),
        ('mistral', 'PD', 'v1', 'baseline', 'baseline'),
        ('mistral', 'PD', 'v2', 'baseline', 'baseline'),
        ('mistral', 'PD', 'v3', 'baseline', 'baseline'),
        ('gemini', 'BoS', 'v1', 'baseline', 'baseline'),
        ('gemini', 'BoS', 'v2', 'baseline', 'baseline'),
        ('gemini', 'BoS', 'v3', 'baseline', 'baseline'),
        ('gemini', 'CKN', 'v1', 'baseline', 'baseline'),
        ('gemini', 'CKN', 'v2', 'baseline', 'baseline'),
        ('gemini', 'CKN', 'v3', 'baseline', 'baseline'),
        ('gemini', 'PD', 'v1', 'baseline', 'baseline'),
        ('gemini', 'PD', 'v2', 'baseline', 'baseline'),
        ('gemini', 'PD', 'v3', 'baseline', 'baseline'),
    ]

    params_neutral_exploiter = [
        ('gpt-3.5', 'BoS', 'v1', 'baseline', 'exploiter'),
        ('gpt-3.5', 'BoS', 'v2', 'baseline', 'exploiter'),
        ('gpt-3.5', 'BoS', 'v3', 'baseline', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v1', 'baseline', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v2', 'baseline', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v3', 'baseline', 'exploiter'),
        ('gpt-3.5', 'PD', 'v1', 'baseline', 'exploiter'),
        ('gpt-3.5', 'PD', 'v2', 'baseline', 'exploiter'),
        ('gpt-3.5', 'PD', 'v3', 'baseline', 'exploiter'),
        ('claude', 'BoS', 'v1', 'baseline', 'exploiter'),
        ('claude', 'BoS', 'v2', 'baseline', 'exploiter'),
        ('claude', 'BoS', 'v3', 'baseline', 'exploiter'),
        ('claude', 'CKN', 'v1', 'baseline', 'exploiter'),
        ('claude', 'CKN', 'v2', 'baseline', 'exploiter'),
        ('claude', 'CKN', 'v3', 'baseline', 'exploiter'),
        ('claude', 'PD', 'v1', 'baseline', 'exploiter'),
        ('claude', 'PD', 'v2', 'baseline', 'exploiter'),
        ('claude', 'PD', 'v3', 'baseline', 'exploiter'),
        ('mistral', 'BoS', 'v1', 'baseline', 'exploiter'),
        ('mistral', 'BoS', 'v2', 'baseline', 'exploiter'),
        ('mistral', 'BoS', 'v3', 'baseline', 'exploiter'),
        ('mistral', 'CKN', 'v1', 'baseline', 'exploiter'),
        ('mistral', 'CKN', 'v2', 'baseline', 'exploiter'),
        ('mistral', 'CKN', 'v3', 'baseline', 'exploiter'),
        ('mistral', 'PD', 'v1', 'baseline', 'exploiter'),
        ('mistral', 'PD', 'v2', 'baseline', 'exploiter'),
        ('mistral', 'PD', 'v3', 'baseline', 'exploiter'),
        ('gemini', 'BoS', 'v1', 'baseline', 'exploiter'),
        ('gemini', 'BoS', 'v2', 'baseline', 'exploiter'),
        ('gemini', 'BoS', 'v3', 'baseline', 'exploiter'),
        ('gemini', 'CKN', 'v1', 'baseline', 'exploiter'),
        ('gemini', 'CKN', 'v2', 'baseline', 'exploiter'),
        ('gemini', 'CKN', 'v3', 'baseline', 'exploiter'),
        ('gemini', 'PD', 'v1', 'baseline', 'exploiter'),
        ('gemini', 'PD', 'v2', 'baseline', 'exploiter'),
        ('gemini', 'PD', 'v3', 'baseline', 'exploiter'),
    ]

    params_grim_neutral = [
        ('gpt-3.5', 'BoS', 'v1', 'GRIM', 'baseline'),
        ('gpt-3.5', 'BoS', 'v2', 'GRIM', 'baseline'),
        ('gpt-3.5', 'BoS', 'v3', 'GRIM', 'baseline'),
        ('gpt-3.5', 'CKN', 'v1', 'GRIM', 'baseline'),
        ('gpt-3.5', 'CKN', 'v2', 'GRIM', 'baseline'),
        ('gpt-3.5', 'CKN', 'v3', 'GRIM', 'baseline'),
        ('gpt-3.5', 'PD', 'v1', 'GRIM', 'baseline'),
        ('gpt-3.5', 'PD', 'v2', 'GRIM', 'baseline'),
        ('gpt-3.5', 'PD', 'v3', 'GRIM', 'baseline'),
        ('claude', 'BoS', 'v1', 'GRIM', 'baseline'),
        ('claude', 'BoS', 'v2', 'GRIM', 'baseline'),
        ('claude', 'BoS', 'v3', 'GRIM', 'baseline'),
        ('claude', 'CKN', 'v1', 'GRIM', 'baseline'),
        ('claude', 'CKN', 'v2', 'GRIM', 'baseline'),
        ('claude', 'CKN', 'v3', 'GRIM', 'baseline'),
        ('claude', 'PD', 'v1', 'GRIM', 'baseline'),
        ('claude', 'PD', 'v2', 'GRIM', 'baseline'),
        ('claude', 'PD', 'v3', 'GRIM', 'baseline'),
        ('mistral', 'BoS', 'v1', 'GRIM', 'baseline'),
        ('mistral', 'BoS', 'v2', 'GRIM', 'baseline'),
        ('mistral', 'BoS', 'v3', 'GRIM', 'baseline'),
        ('mistral', 'CKN', 'v1', 'GRIM', 'baseline'),
        ('mistral', 'CKN', 'v2', 'GRIM', 'baseline'),
        ('mistral', 'CKN', 'v3', 'GRIM', 'baseline'),
        ('mistral', 'PD', 'v1', 'GRIM', 'baseline'),
        ('mistral', 'PD', 'v2', 'GRIM', 'baseline'),
        ('mistral', 'PD', 'v3', 'GRIM', 'baseline'),
        ('gemini', 'BoS', 'v1', 'GRIM', 'baseline'),
        ('gemini', 'BoS', 'v2', 'GRIM', 'baseline'),
        ('gemini', 'BoS', 'v3', 'GRIM', 'baseline'),
        ('gemini', 'CKN', 'v1', 'GRIM', 'baseline'),
        ('gemini', 'CKN', 'v2', 'GRIM', 'baseline'),
        ('gemini', 'CKN', 'v3', 'GRIM', 'baseline'),
        ('gemini', 'PD', 'v1', 'GRIM', 'baseline'),
        ('gemini', 'PD', 'v2', 'GRIM', 'baseline'),
        ('gemini', 'PD', 'v3', 'GRIM', 'baseline'),
    ]
    
    params_grim_exploiter = [
        ('gpt-3.5', 'BoS', 'v1', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'BoS', 'v2', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'BoS', 'v3', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v1', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v2', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'CKN', 'v3', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'PD', 'v1', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'PD', 'v2', 'GRIM', 'exploiter'),
        ('gpt-3.5', 'PD', 'v3', 'GRIM', 'exploiter'),
        ('claude', 'BoS', 'v1', 'GRIM', 'exploiter'),
        ('claude', 'BoS', 'v2', 'GRIM', 'exploiter'),
        ('claude', 'BoS', 'v3', 'GRIM', 'exploiter'),
        ('claude', 'CKN', 'v1', 'GRIM', 'exploiter'),
        ('claude', 'CKN', 'v2', 'GRIM', 'exploiter'),
        ('claude', 'CKN', 'v3', 'GRIM', 'exploiter'),
        ('claude', 'PD', 'v1', 'GRIM', 'exploiter'),
        ('claude', 'PD', 'v2', 'GRIM', 'exploiter'),
        ('claude', 'PD', 'v3', 'GRIM', 'exploiter'),
        ('mistral', 'BoS', 'v1', 'GRIM', 'exploiter'),
        ('mistral', 'BoS', 'v2', 'GRIM', 'exploiter'),
        ('mistral', 'BoS', 'v3', 'GRIM', 'exploiter'),
        ('mistral', 'CKN', 'v1', 'GRIM', 'exploiter'),
        ('mistral', 'CKN', 'v2', 'GRIM', 'exploiter'),
        ('mistral', 'CKN', 'v3', 'GRIM', 'exploiter'),
        ('mistral', 'PD', 'v1', 'GRIM', 'exploiter'),
        ('mistral', 'PD', 'v2', 'GRIM', 'exploiter'),
        ('mistral', 'PD', 'v3', 'GRIM', 'exploiter'),
        ('gemini', 'BoS', 'v1', 'GRIM', 'exploiter'),
        ('gemini', 'BoS', 'v2', 'GRIM', 'exploiter'),
        ('gemini', 'BoS', 'v3', 'GRIM', 'exploiter'),
        ('gemini', 'CKN', 'v1', 'GRIM', 'exploiter'),
        ('gemini', 'CKN', 'v2', 'GRIM', 'exploiter'),
        ('gemini', 'CKN', 'v3', 'GRIM', 'exploiter'),
        ('gemini', 'PD', 'v1', 'GRIM', 'exploiter'),
        ('gemini', 'PD', 'v2', 'GRIM', 'exploiter'),
        ('gemini', 'PD', 'v3', 'GRIM', 'exploiter'),
    ]
    
    # Player 1 uses either GRIM or TFT or neutral policies. TFT is the reference policy.
    # Player 2 uses either neutral or exploiter policies
    # So reference conditions are either TFT vs neutral or TFT vs exploiter
    
    # Evals where player 2 is neutral
    punitiveness_evals_args_list = [
        [
        'programmatic', 
        personas_file, 
        f'config-{model}/config_{game}{version}_{player}-vs-{opponent}.json',
        [f'{base_dir}/{game}/{model}/version{version[-1]}/{player}-{opponent}/{game}v{version[-1]}_{player}-vs-{opponent}-{i}.json' for i in range(10)],
        [f'{base_dir}/{game}/{model}/version{version[-1]}/TFT-baseline/{game}v{version[-1]}_TFT-vs-baseline-{i}.json' for i in range(10)],
        player, 
        f'{game}{version}_{player}-vs-{opponent}'
        ]
        for model, game, version, player, opponent in params_neutral_neutral + params_grim_neutral #params_tft_neutral + 
    ]
    
    # Evals where player 2 is exploiter
    punitiveness_evals_args_list += [
        [
        'programmatic', 
        personas_file, 
        f'config-{model}/config_{game}{version}_{player}-vs-{opponent}.json',
        [f'{base_dir}/{game}/{model}/version{version[-1]}/{player}-{opponent}/{game}v{version[-1]}_{player}-vs-{opponent}-{i}.json' for i in range(10)],
        [f'{base_dir}/{game}/{model}/version{version[-1]}/TFT-exploiter/{game}v{version[-1]}_TFT-vs-exploiter-{i}.json' for i in range(10)],
        player,
        f'{game}{version}_{player}-vs-{opponent}'
        ]
        for model, game, version, player, opponent in params_neutral_exploiter + params_grim_exploiter #params_tft_exploiter +
    ]
    
    # Evals where player 1 is TFT
    exploitability_evals_args_list = [
        [
        'programmatic_exploitability',
        personas_file,
        f'config-{model}/config_{game}{version}_{player}-vs-{opponent}.json',
        [f'{base_dir}/{game}/{model}/version{version[-1]}/{player}-{opponent}/{game}v{version[-1]}_{player}-vs-{opponent}-{i}.json' for i in range(10)],
        [f'{base_dir}/{game}/{model}/version{version[-1]}/TFT-baseline/{game}v{version[-1]}_TFT-vs-baseline-{i}.json' for i in range(10)],
        player,
        f'{game}{version}_{player}-vs-{opponent}'
        ]
        for model, game, version, player, opponent in params_tft_exploiter
    ]
    
    # Evals where player 1 is GRIM
    exploitability_evals_args_list += [
        [
        'programmatic_exploitability',
        personas_file,
        f'config-{model}/config_{game}{version}_{player}-vs-{opponent}.json',
        [f'{base_dir}/{game}/{model}/version{version[-1]}/{player}-{opponent}/{game}v{version[-1]}_{player}-vs-{opponent}-{i}.json' for i in range(10)],
        [f'{base_dir}/{game}/{model}/version{version[-1]}/GRIM-baseline/{game}v{version[-1]}_GRIM-vs-baseline-{i}.json' for i in range(10)],
        player,
        f'{game}{version}_{player}-vs-{opponent}'
        ]
        for model, game, version, player, opponent in params_grim_exploiter
    ]
    
    # Evals where player 1 is neutral
    exploitability_evals_args_list += [
        [
        'programmatic_exploitability',
        personas_file,
        f'config-{model}/config_{game}{version}_{player}-vs-{opponent}.json',
        [f'{base_dir}/{game}/{model}/version{version[-1]}/{player}-{opponent}/{game}v{version[-1]}_{player}-vs-{opponent}-{i}.json' for i in range(10)],
        [f'{base_dir}/{game}/{model}/version{version[-1]}/baseline-baseline/{game}v{version[-1]}_baseline-vs-baseline-{i}.json' for i in range(10)],
        player,
        f'{game}{version}_{player}-vs-{opponent}'
        ]
        for model, game, version, player, opponent in params_neutral_exploiter
    ]
    
    punitiveness_eval_results_paths = parallel_processing(evaluate, punitiveness_evals_args_list)
    print(punitiveness_eval_results_paths)
    plot_punitiveness_bar_chart(punitiveness_eval_results_paths, "punitiveness_bar_chart.png")
        
    exploitability_eval_results_path = parallel_processing(evaluate, exploitability_evals_args_list)
    print(exploitability_eval_results_path)
    plot_exploitability_bar_chart(exploitability_eval_results_path, "exploitability_bar_chart.png")
    
    plot_punitiveness_vs_exploitability(punitiveness_eval_results_paths, exploitability_eval_results_path, "punitiveness_vs_exploitability.png")
    
if __name__ == "__main__":
    main()