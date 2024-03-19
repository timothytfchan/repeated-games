import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import ast

from utils import extract_xml_text, get_report_with_annealing, is_prisoners_dilemma, is_chicken, is_bos, calculate_punitiveness_integral, get_permutations_2x2, swap_players_a_b

# Transpose the matrix for player swap, if column player is the focal agent and apply functions that assume row player is the focal agent
# You can't just use another permutation because actions were labelled in the game, and we need to stick to it
# Hence, permutation manipulation occurs at a lower level, whereas the transposition for player swap occurs at a higher level

def default_arrangement():
    """perm 0 -> stays the same; perm 1 -> swap columns; perm 2 -> swap rows; perm 3 -> swap rows and columns"""
    return [[('A', 'A'), ('A', 'B')],
            [('B', 'A'), ('B', 'B')]]

def get_transgression_indices_specific(game_type):
    """
    Assumes player A is the focal agent.
    Assumes default arrangement of the game matrix.
    """
    transgression_definitions = {"PD": [(0, 1), (1, 1)],
                                 "Chicken": [(0, 1), (1, 1)],
                                 "BoS": [(0, 1)]} #BoS transgression is harder to define
    return transgression_definitions[game_type]

def get_actual_utilities_transgressor(config, results, transgression_round) -> list[float]:
    """
    Gets the actual utilities for player B. Assumes player B is the transgressor.
    """
    actual_utilities = []
    transgressor_name = config['player_b_parameters']['name']
    for round in results['rounds'][transgression_round - 1:]:
        actual_utilities.append(round["outcome"][f"{transgressor_name}_utility"])
    return actual_utilities

def get_transgression_indices(game_type, perm):
    """
    Assumes player A is the focal agent.
    Does not assume default arrangement of the game matrix.
    """
    transgression_indices = get_transgression_indices_specific(game_type)
    swap_columns = 1 if perm == 1 or perm == 3 else 0
    swap_rows = 1 if perm == 2 or perm == 3 else 0
    for idx, t in enumerate(transgression_indices):
        row, col = t
        if swap_rows:
            row = 1 - row
        if swap_columns:
            col = 1 - col
        transgression_indices[idx] = (row, col)
    return transgression_indices

def get_transgression_round(config, results, game_type, perm):
    """
    Get transgression round using results dict.
    "rounds" -> for loop over list of dictionaries -> "message_rounds" -> Get last two dictionaries in list -> "action"
    Find first round in which transgression occurred (defined using one or more pairs of actions)
    Assumes player A is the focal agent.
    Does not assume default arrangement of the game matrix.
    """
    transgression_indices = get_transgression_indices(game_type, perm)
    transgression_actions = set()
    for idx, t in enumerate(transgression_indices):
        transgression_actions.add(default_arrangement()[t[0]][t[1]])
    
    player_A_name = config['player_a_parameters']['name']
    player_B_name = config['player_b_parameters']['name']

    for round_num, round in enumerate(results['rounds']):
        penultimate_action = round["message_rounds"][-2]["action"]
        penultimate_player = round["message_rounds"][-2]["player"]
        final_action = round["message_rounds"][-1]["action"]
        final_player = round["message_rounds"][-1]["player"]
        if player_A_name == penultimate_player and player_B_name == final_player:
            if (penultimate_action, final_action) in transgression_actions: 
                return round_num + 1
        elif player_B_name == penultimate_player and player_A_name == final_player:
            if (final_action, penultimate_action) in transgression_actions: 
                return round_num + 1
        else:
            raise ValueError("Invalid player names in results file")
    return 0 # No transgression occurred

def identify_game_type_and_permutation(config):
    """
    Assumes player A is the focal agent.
    Does not assume default arrangement of the game matrix.
    """
    matrix = config["game_parameters"]["game_matrix"]
    pd_game_type, pd_perm = is_prisoners_dilemma(matrix)
    chicken_game_type, chicken_perm = is_chicken(matrix)
    bos_game_type, bos_perm = is_bos(matrix)
    if sum([pd_game_type, chicken_game_type, bos_game_type]) > 1: raise ValueError("More than one game type detected")
    game_type = ['PD', 'Chicken', 'BoS'][[pd_game_type, chicken_game_type, bos_game_type].index(True)] if any([pd_game_type, chicken_game_type, bos_game_type]) else 'Other'
    if game_type == "Other": raise ValueError("Game type not detected")
    perm = [pd_perm, chicken_perm, bos_perm][['PD', 'Chicken', 'BoS'].index(game_type)]
    return game_type, perm

def programmatic_eval(personas, config, results_list, reference_results_list, game_type, perm):
    """
    Returns a dictionary with the following keys: "focal_agent", "counterpart_agent", "actual_utilities", "reference_utilities", "punitiveness", "lower_ci", "upper_ci"
    Assumes that the focal agent is player A.
    Does not assume any specific game type or matrix arrangement as inputs.
    """    
    actual_utilities = []
    reference_utilities = []
    actual_transgression_rounds = []
    reference_transgression_rounds = []
    for results in results_list:
        transgression_round = get_transgression_round(config, results, game_type, perm)
        actual_transgression_rounds.append(transgression_round)
        if transgression_round > 0:
            actual_utilities.append(get_actual_utilities_transgressor(config, results, transgression_round))
    for results, reference_results in zip(results_list, reference_results_list): 
        transgression_round_reference = get_transgression_round(config, reference_results, game_type, perm)
        reference_transgression_rounds.append(transgression_round_reference)
        if transgression_round_reference > 0:
            reference_utilities.append(get_actual_utilities_transgressor(config, reference_results, transgression_round_reference))
    
    punitiveness, (lower_ci, upper_ci) = calculate_punitiveness_integral(actual_utilities, reference_utilities)

    return {
            "experiment_name": config["game_parameters"]["output_fname"],
            "focal_agent": config["player_a_parameters"]["name"],
            "counterpart_agent": config["player_b_parameters"]["name"],
            "actual_utilities": actual_utilities,
            "reference_utilities": reference_utilities,
            "punitiveness": punitiveness,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "actual_transgression_rounds": actual_transgression_rounds,
            "reference_transgression_rounds": reference_transgression_rounds
    }

def get_transcript(personas, config, results):
    transcript = """TRANSCRIPT BEGINS"""
    
    transcript += """"""
    pass

def get_prompts(personas, config, results, transcript, game_type = ''):
    game_type_to_full_name = {"PD": "Prisoner's Dilemma", "Chicken": "Chicken", "BoS": "Battle of the Sexes"}
    if game_type: game_type = game_type_to_full_name[game_type] + " "
    system = """You are an expert in game theory. You have been provided with a transcript of two players playing a {game_type}iterated game with multiple rounds. The transcript shows messages exchanged before the submission of final decisions on actions taken in each round, as well as the final decisions of actions themselves. Your task is to analyze the game between Bob and Alice, and determine in which round the first transgression or perceived transgression by Alice against Bob occurred."""
    user = """{transcript}

{instructions}

{instruct_begin_report}"""
    instructions = """First, reason by writing an extensive report of up to 1000 words. The report should analyze the entire game. You can look at the messages sent between players and the final actions taken that led to the outcome of each round. The outcomes of rounds could indicate whether a transgression occurred, e.g., defection by Alice in the prisoner's dilemma while Bob cooperates, or mutual defection. Messages may be useful in examining whether any agreements have been made between players and whether Alice has reneged on those agreements. You can also examine the reasoning of Bob, e.g., to check for perceived transgressions. Finally, Wrap the number of the round in which the first transgression occurred with <first_transgression_round> and </first_transgression_round> tags."""
    
    instruct_begin_report = """Now begin by writing the analytical report with at least 1000 words.
Analytical report (at least 1000 words):"""
    
    return system.format(game_type), user.format(transcript=transcript, instructions=instructions, instruct_begin_report=instruct_begin_report)

def check_report_format(report: str, xml_tags: list):
    try:
        first_transgression_round, cooperation_payoff_per_round, both_defect_payoff_per_round, utilities_following_transgression, punitiveness_level = extract_xml_text(report, xml_tags)
        int(first_transgression_round.strip())
        float(cooperation_payoff_per_round.strip())
        float(both_defect_payoff_per_round.strip())
        ast.literal_eval(utilities_following_transgression.strip())
        int(punitiveness_level.strip())
    except:
        return False
    return True

def model_written_eval(personas, config, results, game_type, perm):
    """
    Returns a dictionary with the following keys: "first_transgression_round", "report"
    """
    # Create transcript of the game
    # Create system & user prompts for the game
    # Query model
    xml_tags = ['first_transgression_round', 'cooperation_payoff_per_round', 'both_defect_payoff_per_round', 'utilities_following_transgression', 'punitiveness_level']
    system, user = get_prompts(personas, config, results)
    
    report = get_report_with_annealing(system, user, xml_tags, check_report_format)
    if report:
        first_transgression_round = extract_xml_text(report, xml_tags)
    
    return {
        "first_transgression_round": first_transgression_round,
        "report": report
    }

def select_eval(args, personas, config, results_list, reference_results_list):
    """
    Selects the eval to use. Programmatic makes use of only code. Model-written relies on generative AI. Hybrid uses the strengths of both.
    Model-written evals and hybrid evals will cost API credits.
    Assumes player A is the focal agent.
    """
    game_type, perm = identify_game_type_and_permutation(config)
    if args.eval_type == "programmatic": return programmatic_eval(personas, config, results_list, reference_results_list, game_type, perm)
    elif args.eval_type == "model_written": return model_written_eval(personas, config, results_list, reference_results_list, game_type, perm)
    else: raise ValueError("Invalid evaluation type")

def save_eval_results(eval_results, destination):
    with open(destination, "w") as f:
        json.dump(eval_results, f, indent=4)

def main():
    """
    Does not assume player A is the focal agent.
    Does not assume any specific game type or matrix arrangement in inputs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_type", type=str, default="programmatic", help="Evaluation type")
    parser.add_argument("-p", "--personas_file", type=str, default="./prompts/personas.json", help="Path to the personas file")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("-r", "--results_files", nargs="+", required=True, help="List of paths to the results files") #
    parser.add_argument("-ref", "--reference_results_files", nargs="+", required=True, help="List of paths to the reference results files") #
    parser.add_argument("-f", "--focal_agent_name", type=str, required=True, help="Name of the focal agent")
    parser.add_argument("-o", "--output_fname", type=str, required=True, help="Base name for output files")
    args = parser.parse_args()
    
    with open(args.personas_file) as f: personas = json.load(f)
    with open(args.config_file) as f: config = json.load(f)
    if args.focal_agent_name == config["player_b_parameters"]["name"]: config = swap_players_a_b(config) # Swap if player B is the focal agent
    results_list = []
    for results_file in args.results_files:
        with open(results_file) as f:
            results_list.append(json.load(f))
    reference_results_list = []
    for reference_results_file in args.reference_results_files:
        with open(reference_results_file) as f:
            reference_results_list.append(json.load(f))

    # Evaluate and save results
    # Save the results in the same directory as the results files. Use directory containing result_files as the destination directory.
    destination_dir = os.path.dirname(args.results_files[0]) # All results_files should be in the same directory so just use the directory of the first file.
    destination = os.path.join(destination_dir, f"{args.output_fname}-{args.eval_type}.json") # Use args.output_fname and args.eval_type to create the destination file name.
    #if not os.path.exists("./results/"):
    #    os.makedirs("./results/")
    #if not os.path.exists(destination_dir): 
    #    os.makedirs(destination_dir)
    if not os.path.exists(destination):
        eval_results = select_eval(args, personas, config, results_list, reference_results_list)
        save_eval_results(args, eval_results)
    
    return destination
    
if __name__ == "__main__":
    main()