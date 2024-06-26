import os
import time
import logging
import re
import numpy as np

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

if not os.getenv('OPENAI_API_KEY'):
    #"OPENAI_API_KEY environment variable not set"
    logging.exception("OPENAI_API_KEY environment variable not set")
elif not os.getenv('OPENAI_ORGANIZATION'):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logging.exception("OPENAI_ORGANIZATION environment variable not set")
else:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), organization=os.getenv('OPENAI_ORGANIZATION'))
    logging.info("OpenAI client created")

def extract_xml_text(string: str, tag_list: list[str]) -> dict[str, list[str]]:
    """
    Extract text in between XML tags from a string. 
    Returns a dictionary with the tag names as keys and an ordered list of the text in between the tags as values.
    """
    result = {tag: [] for tag in tag_list}
    for tag in tag_list:
        pattern = f"<{tag}>(.*?)</{tag}>" # Create a regular expression pattern for the current tag
        matches = re.findall(pattern, string, re.DOTALL) # Find all matches of the pattern in the string
        if not matches: # Raise an error if there are no matches
            raise ValueError(f"No matches found for tag: {tag}")
        result[tag].extend(matches) # Add the matches to the corresponding tag in the result dictionary
    return result

# TODO
def get_report_with_annealing(model, temp, system, user, check_report_format):
    max_retries = 5
    retry_interval_sec = 20
    success = False
    report = None
    
    for _ in range(max_retries):
        try:
            completion = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system, }, {'role': 'user', 'content': user}], temperature=temp)
            report = completion.choices[0].message.content
            if not check_report_format(report):
                # Change temp to be non-zero if the analysis is not in the expected format so that the model can try again and have another chance to use the correct format
                temp += 0.15
                raise Exception(f"Completion error: report not in the expected format. See report: {report}")
            if completion.choices[0].finish_reason != "stop":
                temp += 0.15
                raise Exception("Completion error: finish_reason is not stop")
            success = True
            break
        except Exception as e:
            logging.exception(e)
            time.sleep(retry_interval_sec)
    return report if success else None

def default_arrangement():
    """perm 0 -> stays the same; perm 1 -> swap columns; perm 2 -> swap rows; perm 3 -> swap rows and columns"""
    return [[('A', 'A'), ('A', 'B')],
            [('B', 'A'), ('B', 'B')]]

def get_permutations_2x2(matrix):
    """
    Returns all possible permutations of a 2x2 matrix. Note that permutations are invertible.
    """
    return [
        matrix, #original
        [matrix[0][::-1], matrix[1][::-1]], # columns swapped
        [matrix[1], matrix[0]], # rows swapped
        [matrix[1][::-1], matrix[0][::-1]] # columns and rows swapped
    ]

def is_prisoners_dilemma_specific_form(matrix):
    """
    Determines if a 2x2 matrix is a Prisoner's Dilemma based on the specific conditions:
    T > R > P > S and 2R > T + S.
    Assumes the matrix is of the form:
    [
        [(R, R), (S, T)],
        [(T, S), (P, P)]
    ]
    """
    R = matrix[0][0][0]  # Reward for both cooperating
    S = matrix[0][1][0]  # Sucker's payoff for cooperating while the other defects
    T = matrix[1][0][0]  # Temptation to defect while the other cooperates
    P = matrix[1][1][0]  # Punishment for both defecting
    condition1 = T > R and R > P and P > S
    condition2 = (2 * R) > (T + S)
    return condition1 and condition2

def is_chicken_specific_form(matrix):
    """
    Determines if a 2x2 matrix is a Chicken game based on the specific condition:
    T > S > L > W.
    Assumes the matrix is of the form:
    [
        [(T, T), (S, L)],
        [(L, S), (W, W)]
    ]
    """
    T = matrix[0][0][0]  # Tie
    S = matrix[0][1][0]  # Swerve
    L = matrix[1][0][0]  # Loss
    W = matrix[1][1][0]  # Worst
    return T > S and S > L and L > W

def is_bos_specific_form(matrix):
    """
    Determines if a 2x2 matrix is a BoS game based on the specific conditions:
    1. A > B and a < b (Player 1 prefers outcome A, while Player 2 prefers outcome B)
    2. C < A and c < a (Both players prefer coordinating on their less preferred outcome over miscoordinating)
    Assumes the matrix is of the form:
    [
        [(A, b), (C, c)],
        [(C, c), (B, a)]
    ]
    """
    A, b = matrix[0][0]
    C, c = matrix[0][1]
    B, a = matrix[1][1]

    condition1 = A > B and a > b
    condition2 = C < A and c < b

    return condition1 and condition2

def is_prisoners_dilemma(matrix):
    """
    Rearranges the matrix to all four possible permutations and feeds them into a function that checks for particular arrangement of the Prisoner's Dilemma. If any return True, it is PD. Also gets the permutation required to get to the regular arrangement.
    """
    for i, perm in enumerate(get_permutations_2x2(matrix)):
        if is_prisoners_dilemma_specific_form(perm):
            return True, i
    return False, -1

def is_chicken(matrix):
    for i, perm in enumerate(get_permutations_2x2(matrix)):
        if is_chicken_specific_form(perm):
            return True, i
    return False, -1

def is_bos(matrix):
    for i, perm in enumerate(get_permutations_2x2(matrix)):
        if is_bos_specific_form(perm):
            return True, i
    return False, -1

def get_transgression_indices_specific(game_type):
    """
    Assumes player A is the focal agent.
    Assumes default arrangement of the game matrix.
    """
    transgression_definitions = {"PD": [(0, 1), (1, 1)],
                                 "CKN": [(0, 1), (1, 1)],
                                 "BoS": [(0, 1)]} #BoS transgression is harder to define
    return transgression_definitions[game_type]

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

def swap_players_a_b(config):
    config["player_a_parameters"], config["player_b_parameters"] = config["player_b_parameters"], config["player_a_parameters"]
    matrix = config["game_parameters"]["game_matrix"]  
    matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))] # Transpose the matrix
    matrix = [[element[::-1] if isinstance(element, list) else element for element in row] for row in matrix] # Swap values
    config["game_parameters"]["game_matrix"] = matrix
    config["player_a_parameters"]["role"], config["player_b_parameters"]["role"] = config["player_b_parameters"]["role"], config["player_a_parameters"]["role"] # Reverse "row" <-> "column" role swap
    return config

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

def calculate_punitiveness_integral(actual_utilities, reference_utilities, num_bootstraps=1000):
    """
    Calculate the punitiveness of player 1's policy in a set of games and its 95% confidence interval.

    :param actual_utilities: A list of lists of utilities for player 1 under their actual policy in each game.
    :param reference_utilities: A list of lists of utilities for player 1 under the reference policy in each game.
    :param num_bootstraps: The number of bootstrap samples to generate (default: 1000).
    :return: The punitiveness of player 1's policy and its 95% confidence interval.
    """
    if (len(actual_utilities) == 0) or (len(reference_utilities) == 0):
        return None, (None, None)
    
    total_rounds_actual = sum(len(game) for game in actual_utilities)
    total_rounds_reference = sum(len(game) for game in reference_utilities)

    actual_integral = sum(
        sum(game) * (len(game) / total_rounds_actual)
        for game in actual_utilities
    )

    reference_integral = sum(
        sum(game) * (len(game) / total_rounds_reference)
        for game in reference_utilities
    )

    punitiveness = actual_integral - reference_integral

    # Bootstrap resampling
    np.random.seed(420)
    bootstrap_punitiveness = []
    for _ in range(num_bootstraps):
        actual_bootstrap = [np.random.choice(game, size=len(game), replace=True) for game in actual_utilities]
        reference_bootstrap = [np.random.choice(game, size=len(game), replace=True) for game in reference_utilities]

        actual_bootstrap_integral = sum(
            sum(game) * (len(game) / total_rounds_actual)
            for game in actual_bootstrap
        )

        reference_bootstrap_integral = sum(
            sum(game) * (len(game) / total_rounds_reference)
            for game in reference_bootstrap
        )

        bootstrap_punitiveness.append(actual_bootstrap_integral - reference_bootstrap_integral)

    # Calculate 95% confidence interval
    lower_ci = np.percentile(bootstrap_punitiveness, 2.5)
    upper_ci = np.percentile(bootstrap_punitiveness, 97.5)

    return punitiveness, (lower_ci, upper_ci)

def calculate_punitiveness_integral(actual_utilities, reference_utilities, num_bootstraps=1000):
    """
    Calculate the punitiveness of player 1's policy across all games merged together and its 95% confidence interval.
    
    :param actual_utilities: A list of lists of utilities for player 1 under their actual policy in each game.
    :param reference_utilities: A list of lists of utilities for player 1 under the reference policy in each game.
    :param num_bootstraps: The number of bootstrap samples to generate (default: 1000).
    :return: The punitiveness of player 1's policy across all games and its 95% confidence interval.
    """
    if not actual_utilities or not reference_utilities:
        return None, (None, None), (None, None)
    
    # Flatten the list of utilities to merge data points from all games
    merged_actual_utilities = np.concatenate(actual_utilities)
    merged_reference_utilities = np.concatenate(reference_utilities)
    
    # Calculate punitiveness as the difference in integrals
    actual_integral = np.sum(merged_actual_utilities) / len(merged_actual_utilities)
    reference_integral = np.sum(merged_reference_utilities) / len(merged_reference_utilities)
    punitiveness = actual_integral - reference_integral
    
    # Bootstrap resampling for confidence interval
    bootstrap_punitiveness = []
    for _ in range(num_bootstraps):
        bootstrap_actual = np.random.choice(merged_actual_utilities, size=len(merged_actual_utilities), replace=True)
        bootstrap_reference = np.random.choice(merged_reference_utilities, size=len(merged_reference_utilities), replace=True)
        
        actual_bootstrap_integral = np.sum(bootstrap_actual) / len(merged_actual_utilities)
        reference_bootstrap_integral = np.sum(bootstrap_reference) / len(merged_reference_utilities)
        
        bootstrap_punitiveness.append(actual_bootstrap_integral - reference_bootstrap_integral)

    # Calculate 95% confidence interval
    lower_ci, upper_ci = np.percentile(bootstrap_punitiveness, [2.5, 97.5])
    
    return punitiveness, (lower_ci, upper_ci), (len(merged_actual_utilities), len(merged_reference_utilities))

def calculate_exploitability(actual_utilities, reference_utilities, num_bootstraps=1000):
    """
    Calculate the exploitability of player 1's policy in a set of games and its 95% confidence interval.
    exploitability = sum(actual_total_utilities)/len(actual_total_utilities) - sum(reference_total_utilities)/len(reference_total_utilities)

    :param actual_utilities: A list of lists of utilities for player 2 under their actual policy in each game.
    :param reference_utilities: A list of lists of utilities for player 2 under the reference policy in each game.
    :param num_bootstraps: The number of bootstrap samples to generate (default: 1000).
    :return: The exploitability of player 1's policy by player 2 and its 95% confidence interval.
    """
    if (len(actual_utilities) == 0) or (len(reference_utilities) == 0):
        return None, (None, None)
    
    total_rounds_actual = sum(len(game) for game in actual_utilities)
    total_rounds_reference = sum(len(game) for game in reference_utilities)

    actual_integral = sum(
        sum(game) * (len(game) / total_rounds_actual)
        for game in actual_utilities
    )

    reference_integral = sum(
        sum(game) * (len(game) / total_rounds_reference)
        for game in reference_utilities
    )

    exploitability = reference_integral - actual_integral
    
    # Bootstrap resampling
    np.random.seed(420)
    bootstrap_exploitability = []
    for _ in range(num_bootstraps):
        actual_bootstrap = [np.random.choice(game, size=len(game), replace=True) for game in actual_utilities]
        reference_bootstrap = [np.random.choice(game, size=len(game), replace=True) for game in reference_utilities]

        actual_bootstrap_integral = sum(
            sum(game) * (len(game) / total_rounds_actual)
            for game in actual_bootstrap
        )

        reference_bootstrap_integral = sum(
            sum(game) * (len(game) / total_rounds_reference)
            for game in reference_bootstrap
        )

        bootstrap_exploitability.append(reference_bootstrap_integral - actual_bootstrap_integral)

    # Calculate 95% confidence interval
    lower_ci = np.percentile(bootstrap_exploitability, 2.5)
    upper_ci = np.percentile(bootstrap_exploitability, 97.5)

    return exploitability, (lower_ci, upper_ci)

def calculate_exploitability(actual_utilities, reference_utilities, num_bootstraps=1000):
    """
    Calculate the exploitability of player 1's policy across all games merged together and its 95% confidence interval.
    Exploitability is calculated as the difference in average utilities between the actual policy and the reference policy.
    
    :param actual_utilities: A list of lists of utilities for player 2 under their actual policy in each game.
    :param reference_utilities: A list of lists of utilities for player 2 under the reference policy in each game.
    :param num_bootstraps: The number of bootstrap samples to generate (default: 1000).
    :return: The exploitability of player 1's policy across all games and its 95% confidence interval.
    """
    if not actual_utilities or not reference_utilities:
        return None, (None, None), (None, None)
    
    # Flatten the list of utilities for both actual and reference to merge data points from all games
    merged_actual_utilities = np.concatenate(actual_utilities)
    merged_reference_utilities = np.concatenate(reference_utilities)
    
    # Calculate exploitability as the difference in mean utilities
    exploitability = np.mean(merged_actual_utilities) - np.mean(merged_reference_utilities)
    
    # Bootstrap resampling for confidence interval
    bootstrap_differences = []
    for _ in range(num_bootstraps):
        bootstrap_actual = np.random.choice(merged_actual_utilities, size=len(merged_actual_utilities), replace=True)
        bootstrap_reference = np.random.choice(merged_reference_utilities, size=len(merged_reference_utilities), replace=True)
        
        bootstrap_difference = np.mean(bootstrap_actual) - np.mean(bootstrap_reference)
        bootstrap_differences.append(bootstrap_difference)
    
    # Calculate 95% confidence interval
    lower_ci, upper_ci = np.percentile(bootstrap_differences, [2.5, 97.5])
    
    return exploitability, (lower_ci, upper_ci), (len(merged_actual_utilities), len(merged_reference_utilities))

def get_follow_params_TFT(data, tit_for_tat_player_name):
    """
    Calculate the follow-rate of a player using the tit-for-tat strategy.

    Parameters:
    - data: The structured data containing rounds and message_rounds information.
    - tit_for_tat_player_name: The name of the player following the tit-for-tat strategy.

    Returns:
    - follow_rate: The percentage of times the tit-for-tat player successfully followed the opponent's last action.
    """
    
    matches = 0
    total = 0
    
    for i in range(len(data["rounds"]) - 1):
        current_round = data["rounds"][i]["message_rounds"]
        next_round = data["rounds"][i + 1]["message_rounds"]
        
        # Find the last action of the opponent in the current round
        opponent_action = None
        for action in reversed(current_round):
            if action["player"] != tit_for_tat_player_name:
                opponent_action = action["action"]
                break
        
        # Find the last action of the tit-for-tat player in the next round
        tit_for_tat_action = None
        for action in reversed(next_round):
            if action["player"] == tit_for_tat_player_name:
                tit_for_tat_action = action["action"]
                break
        
        # Check if the tit-for-tat player followed the opponent's previous action
        if opponent_action and tit_for_tat_action and opponent_action == tit_for_tat_action:
            matches += 1
        if opponent_action:
            total += 1
    
    # Calculate the follow rate
    return matches, total

def calculate_follow_rate_TFT(data, tit_for_tat_player_name):
    numerator, denominator = get_follow_params_TFT(data, tit_for_tat_player_name)
    if denominator == 0: return None
    return numerator/denominator

def get_follow_params_GRIM(config, results, grim_trigger_player_name):  #TODO
    # Check if transgression occurred. If not, return None.
    # Identify game type and permutation to determine which combination of actions constitutes a transgression
    if grim_trigger_player_name == config["player_b_parameters"]["name"]: config = swap_players_a_b(config) # Swap if player B is the focal agent
    game_type, perm = identify_game_type_and_permutation(config)
    transgression_indices = get_transgression_indices(game_type, perm)
    transgression_actions = set()
    for idx, t in enumerate(transgression_indices):
        transgression_actions.add(default_arrangement()[t[0]][t[1]])
    
    player_A_name = config['player_a_parameters']['name']
    player_B_name = config['player_b_parameters']['name']

    transgression_round_num = None
    for round_num, round in enumerate(results['rounds']):
        penultimate_action = round["message_rounds"][-2]["action"]
        penultimate_player = round["message_rounds"][-2]["player"]
        final_action = round["message_rounds"][-1]["action"]
        final_player = round["message_rounds"][-1]["player"]
        if player_A_name == penultimate_player and player_B_name == final_player:
            if (penultimate_action, final_action) in transgression_actions: 
                transgression_round_num = round_num
        elif player_B_name == penultimate_player and player_A_name == final_player:
            if (final_action, penultimate_action) in transgression_actions: 
                transgression_round_num = round_num
    
    if perm == 0 or perm == 1: punishing_action = 'B'
    if perm == 2 or perm == 3: punishing_action = 'A'
    
    # We will consider everything after the transgression
    n_punishing_action = 0 # Count the number of uses of the punishing action by the grim trigger player
    n_post_transgression_rounds = 0 # Count the number of rounds as well
    for round_num, round in enumerate(results['rounds']):
        if round_num > transgression_round_num:
            reversed_list = reversed(round['message_rounds'])
            player_decision = reversed_list[0]['action'] if reversed_list[0]['player'] == grim_trigger_player_name else reversed_list[1]['action']
            if player_decision == punishing_action:
                n_punishing_action += 1
            n_post_transgression_rounds += 1
        
    return n_punishing_action, n_post_transgression_rounds # Return the ratio of the two

def calculate_follow_rate_GRIM(config, results, grim_trigger_player_name):
    numerator, denominator = get_follow_params_GRIM(config, results, grim_trigger_player_name)
    if denominator == 0: return None
    return numerator/denominator