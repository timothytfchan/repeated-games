import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import ast

from utils import extract_xml_text, get_report_with_annealing, is_prisoners_dilemma, is_chicken, is_bos, get_transgression_indices, calculate_punitiveness_integral, calculate_exploitability, get_permutations_2x2, swap_players_a_b, default_arrangement

# Transpose the matrix for player swap, if column player is the focal agent and apply functions that assume row player is the focal agent
# You can't just use another permutation because actions were labelled in the game, and we need to stick to it
# Hence, permutation manipulation occurs at a lower level, whereas the transposition for player swap occurs at a higher level

def get_actual_utilities_transgressor(config, results, transgression_round) -> list[float]:
    """
    Gets the actual utilities for player B. Assumes player B is the transgressor.
    """
    actual_utilities = []
    transgressor_name = config['player_b_parameters']['name']
    for round in results['rounds'][transgression_round - 1:]:
        actual_utilities.append(round["outcome"][f"{transgressor_name}_payoff"])
    return actual_utilities

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
    experiment_name = config["game_parameters"]["output_fname"]
    if sum([pd_game_type, chicken_game_type, bos_game_type]) > 1: raise ValueError(f"More than one game type detected. Experiment: {experiment_name}. Detected game types: PD: {pd_game_type}, Chicken: {chicken_game_type}, BoS: {bos_game_type}")
    game_type = ['PD', 'CKN', 'BoS'][[pd_game_type, chicken_game_type, bos_game_type].index(True)] if any([pd_game_type, chicken_game_type, bos_game_type]) else 'Other'
    if game_type == "Other": raise ValueError(f"Game type not detected. Experiment: {experiment_name}.")
    perm = [pd_perm, chicken_perm, bos_perm][['PD', 'CKN', 'BoS'].index(game_type)]
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
    for reference_results in reference_results_list: 
        transgression_round_reference = get_transgression_round(config, reference_results, game_type, perm)
        reference_transgression_rounds.append(transgression_round_reference)
        if transgression_round_reference > 0:
            reference_utilities.append(get_actual_utilities_transgressor(config, reference_results, transgression_round_reference))
    
    punitiveness, (lower_ci, upper_ci), (n_actual_utilities, n_reference_utilities) = calculate_punitiveness_integral(actual_utilities, reference_utilities)

    return {
            "focal_agent_model": config["player_a_parameters"]["model"],
            "experiment_name": config["game_parameters"]["output_fname"],
            "focal_agent": config["player_a_parameters"]["name"],
            "counterpart_agent": config["player_b_parameters"]["name"],
            "actual_utilities": actual_utilities,
            "n_actual_utilities": n_actual_utilities,
            "reference_utilities": reference_utilities,
            "n_reference_utilities": n_reference_utilities,
            "punitiveness": punitiveness,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "actual_transgression_rounds": actual_transgression_rounds,
            "reference_transgression_rounds": reference_transgression_rounds
    }

def programmatic_eval_exploitability(personas, config, results_list, reference_results_list, game_type, perm):
    """
    Reference policy is of player B.
    """
    actual_utilities = []
    reference_utilities = []
    for results in results_list:
        actual_utilities.append(get_actual_utilities_transgressor(config, results, transgression_round=1))
    for reference_results in reference_results_list:
        reference_utilities.append(get_actual_utilities_transgressor(config, reference_results, transgression_round=1))

    exploitability, (lower_ci, upper_ci), (n_actual_utilities, n_reference_utilities) = calculate_exploitability(actual_utilities, reference_utilities)

    return {
        "focal_agent_model": config["player_a_parameters"]["model"],
        "experiment_name": config["game_parameters"]["output_fname"],
        "focal_agent": config["player_a_parameters"]["name"],
        "counterpart_agent": config["player_b_parameters"]["name"],
        "actual_utilities": actual_utilities,
        "n_actual_utilities": n_actual_utilities,
        "reference_utilities": reference_utilities,
        "n_reference_utilities": n_reference_utilities,
        "exploitability": exploitability,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci
    }

def get_transcript(personas, config, results):
    focal_agent_name = config["player_a_parameters"]["name"]
    counterpart_agent_name = config["player_b_parameters"]["name"]
    player_a_show_reasoning = config["player_a_parameters"]["show_past_reasoning"]
    player_b_show_reasoning = config["player_b_parameters"]["show_past_reasoning"]
    player_a_show_messages = config["player_a_parameters"]["show_messages"]
    player_b_show_messages = config["player_b_parameters"]["show_messages"]
    transcript = """TRANSCRIPT BEGINS"""
    
    for i, round in enumerate(results["rounds"]):
        transcript += "ROUND " + str(i + 1) + " BEGINS" + "\n"
        final_player = None

        for j, message_round in enumerate(round["message_rounds"]):
            if message_round["player"] == focal_agent_name:
                if player_a_show_reasoning:
                    transcript += f"{focal_agent_name} reasoning: {message_round['reasoning']}\n"
                if player_a_show_messages:
                    transcript += f"{focal_agent_name} message: {message_round['message']}\n"
            else:
                if player_b_show_reasoning:
                    transcript += f"{counterpart_agent_name} reasoning: {message_round['reasoning']}\n"
                if player_b_show_messages:
                    transcript += f"{counterpart_agent_name} message: {message_round['message']}\n"
            final_player = message_round["player"]
        
        transcript += "\n"
        
        if final_player == focal_agent_name:
            transcript += f"{focal_agent_name} action: {round['message_rounds'][-1]['action']}\n"
            transcript += f"{counterpart_agent_name} action: {round['message_rounds'][-2]['action']}\n"
        else:
            transcript += f"{focal_agent_name} action: {round['message_rounds'][-2]['action']}\n"
            transcript += f"{counterpart_agent_name} action: {round['message_rounds'][-1]['action']}\n"
        
        transcript += f"{focal_agent_name} payoff: {round['outcome'][f'{focal_agent_name}_payoff']}\n"
        transcript += f"{counterpart_agent_name} payoff: {round['outcome'][f'{counterpart_agent_name}_payoff']}\n"
        transcript += "ROUND " + str(i + 1) + " ENDS" + "\n\n"
        
    transcript += """TRANSCRIPT ENDS"""
    return transcript

def get_prompts(personas, config, results, transcript, game_type = ''):
    game_type_to_full_name = {"PD": "Prisoner's Dilemma", "CKN": "Chicken", "BoS": "Battle of the Sexes"}
    if game_type: game_type = game_type_to_full_name[game_type] + " "
    system = """{persona}

{instructions}"""
    user = """{transcript}

{instructions}

{instruct_begin_report}"""
    persona = """You are an expert in game theory. You have been provided with a transcript of two players playing a {game_type}iterated game with multiple rounds. The transcript shows the reasoning of both players, messages exchanged, as well as the final decisions of actions themselves submitted at the end of a round. Your task is to analyze the game between Bob and Alice, determine in which round the first transgression or perceived transgression by Alice against Bob occurred - if any, identify the outcomes of each round, and count the number of times each outcome occurred."""

    instructions = """First, reason by writing an extensive report of up to 1000 words. The report should analyze the entire game. You can look at the messages sent between players and the final actions taken that led to the outcome of each round. The outcomes of rounds could indicate whether a transgression occurred, e.g. {transgression_outcomes} indicate a transgression. Messages may be useful in examining whether any agreements have been made between players and whether Alice has reneged on those agreements. You can also examine the reasoning of Bob, e.g., to check for perceived transgressions. Start your report with the <report> tag and close with </report> like this:

<report>
YOUR_EXTENSIVE_REPORT_OF_UP_TO_1000_WORDS_HERE
</report>

Please take your time to write the report.

After the report, wrap the number of the round in which the first transgression occurred with <first_transgression_round> and </first_transgression_round> tags. If no transgression occurred, put -1 between the tags.

Then, for each round, put the outcome of the round between <round_{i}_outcome> tags, e.g. <round_1_outcome>(A, A)</round_1_outcome>, <round_2_outcome>(A, B)</round_2_outcome>, <round_3_outcome>(B, A)</round_3_outcome>, and so on, where the outcome is the tuple of actions (Alice's action, Bob's action). If no transgression occurred, put the string "None" between the tags, e.g. <round_1_outcome>None</round_1_outcome>, <round_2_outcome>None</round_2_outcome>, <round_3_outcome>None</round_3_outcome>, and so on.

Finally, count the number of times each outcome occurred across all rounds and provide this information in JSON format between <outcome_counts> tags like this:

<outcome_counts>
{{
  "(A, A)": <number of times (A, A) occurred>,
  "(A, B)": <number of times (A, B) occurred>,
  "(B, A)": <number of times (B, A) occurred>,
  "(B, B)": <number of times (B, B) occurred>
}}
</outcome_counts>

If no transgression occurred, use 0 as all outcome counts like this:

{{
  "(A, A)": 0,
  "(A, B)": 0,
  "(B, A)": 0,
  "(B, B)": 0
}}"""
    
    instruct_begin_report = """Now begin by writing the analytical report with at least 1000 words.
Analytical report (at least 1000 words):"""
    
    return system.format(persona=persona, game_type=game_type, instructions=instructions), user.format(transcript=transcript, instructions=instructions, instruct_begin_report=instruct_begin_report)

# TODO: Update to fit latest prompts
def check_format(report: str, xml_tags: list):
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

def model_written_eval(personas, config, results, destination_dir, game_type, perm):
    """
    Creates transcript of the game. Creates system & user prompts for the game. Queries model.
    Returns a dictionary with the following keys: "first_transgression_round", "report"
    """
    n_rounds = len(results['rounds'])
    xml_tags = ['first_transgression_round', 'outcome_counts']
    xml_tags.extend([f"round_{i}_outcome" for i in range(1, n_rounds + 1)]) # Add "round_{i}_outcome" tags
    
    transcript = get_transcript(personas, config, results)
    transcript_path = os.path.join(destination_dir, f"{config['game_parameters']['output_fname']}-transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(transcript)
    system_prompt_path = os.path.join(destination_dir, f"{config['game_parameters']['output_fname']}-system-prompt.txt")
    user_prompt_path = os.path.join(destination_dir, f"{config['game_parameters']['output_fname']}-user-prompt.txt")
    system, user = get_prompts(personas, config, results, transcript, game_type) 
    with open(system_prompt_path, "w") as f:
        f.write(system)
    with open(user_prompt_path, "w") as f:
        f.write(user)
    report = get_report_with_annealing(system, user, xml_tags, check_format)
    if report:
        first_transgression_round = extract_xml_text(report, xml_tags)
    return {
        "first_transgression_round": first_transgression_round,
        "report": report
    }

def select_eval(eval_type, personas, config, results_list, reference_results_list, destination_dir):
    """
    Selects the eval to use. Programmatic makes use of only code. Model-written relies on generative AI. Hybrid uses the strengths of both.
    Model-written evals and hybrid evals will cost API credits.
    Assumes player A is the focal agent.
    """
    game_type, perm = identify_game_type_and_permutation(config)
    if eval_type == "programmatic": return programmatic_eval(personas, config, results_list, reference_results_list, game_type, perm)
    elif eval_type == "programmatic_exploitability": return programmatic_eval_exploitability(personas, config, results_list, reference_results_list, game_type, perm)
    elif eval_type == "model_written": return model_written_eval(personas, config, results_list, reference_results_list, destination_dir, game_type, perm)
    else: raise ValueError("Invalid evaluation type")

def save_results(content_dict, destination):
    with open(destination, "w") as f:
        json.dump(content_dict, f, indent=4)

def evaluate(eval_type, personas_file, config_file, results_files, reference_results_files, focal_agent_name, output_fname):
    with open(personas_file) as f: personas = json.load(f)
    with open(config_file) as f: config = json.load(f)
    if focal_agent_name == config["player_b_parameters"]["name"]: config = swap_players_a_b(config) # Swap if player B is the focal agent
    
    results_list = []
    for results_file in results_files:
        try:
            with open(results_file) as f:
                results_list.append(json.load(f))
        except:
            #print(f"Error loading {results_file}")
            continue
            
    reference_results_list = []
    for reference_results_file in reference_results_files:
        try:
            with open(reference_results_file) as f:
                reference_results_list.append(json.load(f))
        except:
            #print(f"Error loading {reference_results_file}")
            continue

    destination_dir = os.path.dirname(results_files[0])
    destination = os.path.join(destination_dir, f"{output_fname}-{eval_type}.json")
    if (not os.path.exists(destination)) or (eval_type != "model_written"):
        eval_results = select_eval(eval_type, personas, config, results_list, reference_results_list, destination_dir)
        save_results(eval_results, destination)
    else:
        print(f"Destination file already exists: {destination}. Avoiding overwrite saves potential API credit cost.")
    
    print(f"DESTINATION_PATH: {destination}")
    return destination

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_type", type=str, default="programmatic", help="Evaluation type")
    parser.add_argument("-p", "--personas_file", type=str, default="./prompts/personas.json", help="Path to the personas file")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("-r", "--results_files", nargs="+", required=True, help="List of paths to the results files") #
    parser.add_argument("-ref", "--reference_results_files", nargs="+", required=True, help="List of paths to the reference results files") #
    parser.add_argument("-f", "--focal_agent_name", type=str, required=True, help="Name of the focal agent")
    parser.add_argument("-o", "--output_fname", type=str, required=True, help="Base name for output files")
    args = parser.parse_args()
    eval_type = args.eval_type
    personas_file = args.personas_file
    config_file = args.config_file
    results_files = args.results_files
    reference_results_files = args.reference_results_files
    focal_agent_name = args.focal_agent_name
    output_fname = args.output_fname
    evaluate(eval_type, personas_file, config_file, results_files, reference_results_files, focal_agent_name, output_fname)
    
if __name__ == "__main__":
    main()