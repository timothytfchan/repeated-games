import os
import argparse
from copy import deepcopy
import logging
import json
import string
import re
import numpy as np
import pandas as pd
import openai
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), organization=os.getenv('OPENAI_ORGANIZATION'))
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Function calling hierarchy:
main() calls run_game() multiple times in parallel and initializes objects of class Policy
run_game() calls run_round() and update_results()
run_round() calls query_policy() and results_to_history_str()
results_to_history_str() goes into query_policy()
query_policy() calls get_response()
get_response() calls get_response_openai()
get_response_openai() uses get_prompts()
get_prompts() uses matrix_to_str()
"""

class Policy:
	def __init__(self, model, temperature, player_name, opponent_name, role, persona, strategic_reasoning=False, sys_prompt_substitutions={}, show_past_reasoning=True, show_messages = True, show_intended_actions=True):
		self.player_name = player_name 
		self.opponent_name = opponent_name # for prompt construction
		self.role = role # for determining whether the player acts on rows or columns of matrix
		self.persona = persona # for selecting data from csv file
		self.strategic_reasoning = strategic_reasoning # whether to include optional strategic reasoning
		self.sys_prompt_substitutions = sys_prompt_substitutions # for inserting more expressions into sys prompt (values replace keys in curly braces within template str)
		self.model = model
		self.temperature = temperature
		self.show_past_reasoning = show_past_reasoning
		self.show_messages = show_messages
		self.show_intended_actions = show_intended_actions
		self.most_recent_action = None

	def query_policy(self, history_str, message_round, labels_row, labels_column, end_prob, matrix, total_message_rounds, **kwargs):
		output = get_response(self.player_name, self.opponent_name, self.role, self.persona, end_prob, history_str, matrix, labels_row, labels_column, message_round, total_message_rounds, strategic_reasoning=self.strategic_reasoning, sys_prompt_substitutions = self.sys_prompt_substitutions, model=self.model, temperature=self.temperature, **kwargs)
		self.most_recent_action = output['action']
		return output

# Replacer helper function for inserting values for custom keys
# Used instead of .format because there might be more key-value pairs than needed
# Outer function to create a closure
def create_replacer(values_dict):
    # Inner function to replace placeholders
    def replace_placeholder(match):
        key = match.group(1)
        return str(values_dict.get(key, ""))
    return replace_placeholder

def always_defect():
	answer = '{"reasoning": "", "message": "","action": "D"}'
	return json.loads(answer)

def get_strategic_reasoning_prompt(examples_to_include=('simultaneous_22.txt')):
	files = os.listdir('./prompts/strategic_reasoning')
	prompt = 'Below are examples of strategic reasoning from other games. You should use similar reasoning patterns in making your own decision.'
	for fname in files:
		if fname in examples_to_include:
			with open(os.path.join('./prompts/strategic_reasoning', fname)) as f:
				text = f.read()
			prompt += f'\n{text}'
	return prompt

def matrix_to_str(matrix, labels_row, labels_column, player_name, opponent_name, role):
	if role == 'row':
		row = player_name
		column = opponent_name
	elif role == 'column':
		row = opponent_name
		column = player_name

	m, n = len(matrix), len(matrix[0])
	matrix_str = ''
	for i in range(m):
		row_label = labels_row[i]
		for j in range(n):
			column_label = labels_column[j]
			u_row = matrix[i][j][0]
			u_column = matrix[i][j][1]
			outcome_str = f'\nIf {row} plays {row_label} and {column} plays {column_label}, then {row} gets {u_row} and {column} gets {u_column}.'
			matrix_str += outcome_str
	return matrix_str

# Constructs prompts from the CSV and TXT files
def get_prompts(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round, total_message_rounds, strategic_reasoning, sys_prompt_substitutions = {}):
    # Construct system prompt
	if strategic_reasoning:
		sys_prompt_substitutions["strategic_reasoning_prompt"] = get_strategic_reasoning_prompt()
	
	if role == "row":
		labels = labels_row
		row, column = player_name, opponent_name
	elif role == "column":
		labels = labels_column
		row, column = opponent_name, player_name
	matrix_str = matrix_to_str(matrix, labels_row, labels_column, player_name, opponent_name, role)
	
	labels = '{' + ', '.join(labels) + '}'

	# Include any values that have not been added
	sys_prompt_substitutions['opponent_name'] = opponent_name
	sys_prompt_substitutions['labels'] = labels

	#Set up replacer
	pattern = r'\{([^}]*)\}'
	replacer = create_replacer(sys_prompt_substitutions)

	# import prompt/personas.json and get the intro, add_context, and output_schema values for the persona
	with open('./prompts/personas.json', 'r') as file:
		data = json.load(file)
	intro = data[persona]['intro']
	intro = re.sub(pattern, replacer, intro)

	add_context = data[persona]['add_context']
	add_context = re.sub(pattern, replacer, add_context)

	output_schema = data[persona]['output_schema']
	# Fill in placeholders of every value in output_schema dictionary
	for key, val in output_schema.items():
		output_schema[key] = re.sub(pattern, replacer, val)
	output_schema = json.dumps(output_schema, indent=2)

	system = ""
	system += f"{intro} " if intro else ""
	system += f"Your name is {player_name}, and the agent you are playing against is named {opponent_name}. At each time-step, you and {opponent_name} will exchange messages. {total_message_rounds} messages in total will be exchanged, with the first speaker being chosen at random in each round. In each round, you and {opponent_name} will simultaneously choose your actions. Your available actions are labeled {labels}. The game has a {end_prob}% chance of ending after every time-step. Here are the payoffs from the payoff matrix:\n{matrix_str}.\n\nBefore each time-step, you will be shown a summary of the history of play thus far."
	system += f" {add_context} " if add_context else ""
	system += f"You should return a JSON with the following format (key names should be in double quotes; you must always fill out every field - but you can respond with an empty string if you there are no more message rounds):\n\n{output_schema}"

	# Construct user prompt
	if message_round > total_message_rounds:
		final_message_round = "There are no more message rounds."
	elif message_round in (total_message_rounds-1, total_message_rounds):
		final_message_round = f"It is message round {message_round} of {total_message_rounds}. This is your last chance to send a message."
	else:
		final_message_round = f"It is message round {message_round} of {total_message_rounds}."
	user = f"Here is a summary of the history of play thus far:\n{history_str}.\n\n{final_message_round} Now please output a correctly-formatted JSON:"

	#print("SYSTEM: ", system, '\n\n', 'USER: ', user)
	return system, user

# Sends request and gets JSON response from the OpenAI API. In the future, we may want to add get_response_anthropic etc.
def get_response_openai(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round=None, total_message_rounds=None, strategic_reasoning=False, sys_prompt_substitutions={}, model='gpt-4-32k', temperature = 1.0, **kwargs):
	system, user = get_prompts(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round, total_message_rounds, strategic_reasoning=strategic_reasoning, sys_prompt_substitutions=sys_prompt_substitutions)
	max_retries = 5
	retry_interval_sec = 20
	answer_dict = {}

	# import prompt/personas.json and get the output_schema values for the persona
	with open('./prompts/personas.json', 'r') as file:
		data = json.load(file)
	json_schema = data[persona]['output_schema']
	#print(json_schema)
	
	if model == 'gpt-4-1106-preview':
		kwargs['response_format']={ "type": "json_object" }
 
	for _ in range(max_retries):
		try:
			completion = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system, }, {'role': 'user', 'content': user}], max_tokens=800, temperature=temperature, **kwargs)
			# Not valid JSON if there are too many tokens and it gets cut off
			if completion.choices[0].finish_reason != "stop":
				raise Exception("Completion error: finish_reason is not stop")
			answer = completion.choices[0].message.content
			answer_dict = json.loads(answer)

			# Confirm that it follows the JSON schema we are interested in
			if set(json_schema.keys()) != set(answer_dict.keys()):
				raise Exception(f"JSON returned with missing key")
			break
		except (
			openai.RateLimitError,
			openai.error.ServiceUnavailableError,
			openai.error.APIError,
			openai.error.APIConnectionError,
			openai.error.Timeout,
			openai.error.TryAgain,
			openai.error.OpenAIError,) as e:
			logging.exception(e)
			time.sleep(retry_interval_sec)
		except Exception as e:
			logging.exception(e)
			time.sleep(retry_interval_sec)
	
	if not answer_dict:
		# If it does not succeed after multiple retries, consider this a failed game and log it
		logging.error('Failed to get response from OpenAI API')
 
	return answer_dict

def get_response(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round=None, total_message_rounds=None, strategic_reasoning=False, sys_prompt_substitutions={}, model='gpt-4-32k', temperature=1.0, **kwargs):
    if persona == "always_defect":
        return always_defect()
    
	# if none of the above
    return get_response_openai(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round=message_round, total_message_rounds=total_message_rounds, strategic_reasoning=strategic_reasoning, sys_prompt_substitutions=sys_prompt_substitutions, model=model, temperature = temperature, **kwargs)

def results_to_history_str(policy, names_tuple, results, round_outputs=None):
	history_str = ''
	player_a, player_b = names_tuple[0], names_tuple[1]
	show_past_reasoning = policy.show_past_reasoning
	show_messages = policy.show_messages
	show_intended_actions = policy.show_intended_actions
	current_player = policy.player_name
	# Round level: Summarize previous rounds using results
	for i, round_ in enumerate(results['rounds']):
		history_str += f'ROUND {i+1}:\n'
		# Message round level: Summarize any combination of {reasoning, messages, actions}
		for j, message_round in enumerate(round_['message_rounds']):
			if show_past_reasoning and (message_round['player'] == current_player):
				history_str += f"Your reasoning: {message_round['reasoning']}\n"
			
			if show_messages and (message_round['player'] == current_player):
				history_str += f"Your message: {message_round['message']}\n"

			if show_messages and (message_round['player'] != current_player):
				history_str += f"Other party's message: {message_round['message']}\n"

			if show_intended_actions and (message_round['player'] == current_player):
				history_str += f"Your intended action at t={j+1} within this round: {message_round['action']}\n"

			history_str += "\n"
		
			# Summarize final actions and round outcomes
			if j+1 == len(round_['message_rounds']):
				your_payoff = round_['outcome'][f'{player_a}_payoff'] if current_player == player_a else round_['outcome'][f'{player_b}_payoff']
				your_final_action = round_['message_rounds'][-1]['action'] if round_['message_rounds'][-1]['player'] == current_player else round_['message_rounds'][-2]['action']

				other_payoff = round_['outcome'][f'{player_a}_payoff'] if current_player != player_a else round_['outcome'][f'{player_b}_payoff']
				other_final_action = round_['message_rounds'][-1]['action'] if round_['message_rounds'][-1]['player'] != current_player else round_['message_rounds'][-2]['action']
	
				history_str += f"OUTCOME OF ROUND {i+1}: At the end of the round, you chose {your_final_action}. On the other hand, the other party chose {other_final_action}. This resulted in a payoff of {your_payoff} for you and a payoff of {other_payoff} for the other party.\n\nBEGIN NEW ROUND\n"
		
		history_str += "\n"
	
	# Summarize the new round so far, if there is info
	if round_outputs:
		history_str += f"CURRENT ROUND:\n"
		for j, (player, message_round) in enumerate(round_outputs):
			reasoning = message_round['reasoning']
			message = message_round['message']
			action = message_round['action']
			if player == current_player:
				history_str += f"Your reasoning: {reasoning}\n"
			
			if player == current_player:
				history_str += f"Your message: {message}\n"

			if player != current_player:
				history_str += f"Other party's message: {message}\n"

			if player == current_player:
				history_str += f"Your intended action at t={j+1} within this round: {action}\n"
	
	return history_str.strip()

def run_round(policy_dict, names_tuple, matrix, labels_row, labels_column, results_dict, end_prob, total_message_rounds):
	# Randomly choose who speaks first, then create tuple of speaker indices that alternate
	first_speaker_index = np.random.choice(2)
	speaker_indices = (lambda n, x: tuple(x if i % 2 == 0 else 1 - x for i in range(n)))(total_message_rounds, first_speaker_index)	
	
	# Conduct message exchange and get final action from player who sends the final message
	round_outputs = []

	for ix, speaker_index in enumerate(speaker_indices):
		speaker, non_speaker = names_tuple[speaker_index], names_tuple[1-speaker_index]
		policy = policy_dict[speaker]
		history_str = results_to_history_str(policy, names_tuple, results_dict, round_outputs)
		round_output = policy.query_policy(history_str, ix+1, labels_row, labels_column, end_prob, matrix, total_message_rounds) # LLM thinks and speaks here
		round_outputs.append((speaker, round_output))
		
	# In the case where total_message_rounds is zero, no messages are exchanged, both players only select actions (note: to avoid messages, message_round parameter is set to > total_message_rounds).
	if total_message_rounds == 0:
		speaker_index = first_speaker_index
		speaker, non_speaker = names_tuple[speaker_index], names_tuple[1-speaker_index]
		policy = policy_dict[speaker]
		history_str = results_to_history_str(policy, names_tuple, results_dict, round_outputs, False)
		round_output = policy.query_policy(history_str, total_message_rounds+1, labels_row, labels_column, end_prob, matrix, total_message_rounds) # LLM thinks and speaks here
		round_outputs.append((speaker, round_output))
 
	# Get final action from the player who didn't get to send the final message (if there are message rounds) and the second player (if there are no message rounds)
	last_index = 1 - speaker_index
	speaker, non_speaker = names_tuple[last_index], names_tuple[1-last_index]
	policy = policy_dict[speaker]
	history_str = results_to_history_str(policy, names_tuple, results_dict, round_outputs)
	round_output = policy.query_policy(history_str, total_message_rounds + 1, labels_row, labels_column, end_prob, matrix, total_message_rounds)
	round_outputs.append((speaker, round_output))

	return round_outputs

def update_results(results_dict, round_outputs, round_idx, policy_dict, payoff_dict, names_tuple):
	"""
	Update summary statistics for game using latest round. Example of JSON below. `player_a` and `player_b` are replaced with the names of players.
	-----------------------
	results_dict = {
		'summary': {
			'player_a_total_utility': <value>,
			'player_b_total_utility': <value>,
			'player_a_avg_utility': <value>,
			'player_b_avg_utility': <value>
		},
		'rounds': [
			{
				'message_rounds': [
					{'player': <player>, 'reasoning': <reasoning>, 'message': <message>, 'action': <action>},
					...
				],
				'outcome': {
					'player_a_payoff': <value>,
					'player_b_payoff': <value>
				}
			},
			...
		]
	}
	-----------------------
	As results_dict['rounds'] and results_dict['rounds'][i]['message_rounds'] are lists, rounds and message_rounds have an implicit order.
	"""

	player_a, player_b = names_tuple[0], names_tuple[1]
	player_a_action, player_b_action  = policy_dict[player_a].most_recent_action, policy_dict[player_b].most_recent_action
	player_a_payoff, player_b_payoff = payoff_dict[f'{player_a_action},{player_b_action}']
	results_dict['summary'][f'{player_a}_total_utility'] += player_a_payoff
	results_dict['summary'][f'{player_b}_total_utility'] += player_b_payoff
	results_dict['summary'][f'{player_a}_avg_utility'] = results_dict['summary'][f'{player_a}_total_utility'] / (round_idx+1)
	results_dict['summary'][f'{player_b}_avg_utility'] = results_dict['summary'][f'{player_b}_total_utility'] / (round_idx+1)
	
	results_dict['rounds'].append({})
	results_dict['rounds'][-1]['message_rounds'] = []
	results_dict['rounds'][-1]['outcome'] = {f'{player_a}_payoff': player_a_payoff, f'{player_b}_payoff': player_b_payoff}
	for k, (player, message_round) in enumerate(round_outputs):
		reasoning = message_round['reasoning']
		message = message_round['message']
		action = message_round['action']
		results_dict['rounds'][-1]['message_rounds'].append({'player': player, 'reasoning': reasoning, 'message': message, 'action': action})

	return results_dict

def run_game(player_a_policy, player_b_policy, end_prob: float, fname: str, matrix: list[list], max_game_rounds: int = 20, total_message_rounds: int = 3):
	# Check if runs-{today_date} directory exists, if not, create it
	today_date = time.strftime("%Y-%m-%d")
	output_dir = f"results/runs-{today_date}"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	try:
		# Check errors and if passed, make deep copies of policies
		if (not isinstance(player_a_policy, Policy)) or (not isinstance(player_b_policy, Policy)):
			raise TypeError('Argument is not an instance of the Policy class')
		if player_a_policy.player_name == player_b_policy.player_name:
			raise ValueError('Players cannot have the same name')
		if fname == "":
			raise ValueError('Filename cannot be empty')
		if not (0 <= end_prob <= 1):
			raise ValueError('Probability of termination is out of range')

		player_a_policy = deepcopy(player_a_policy)
		player_b_policy = deepcopy(player_b_policy)
		
		# Initialize values and dictionaries to store results and values
		results_dict = {'summary': {}, 'rounds': []}
		results_dict['summary'][f'{player_a_policy.player_name}_total_utility'] = 0.
		results_dict['summary'][f'{player_b_policy.player_name}_total_utility'] = 0.
		policy_dict = {player_a_policy.player_name: player_a_policy, 
						player_b_policy.player_name: player_b_policy}
		names_tuple = (player_a_policy.player_name, player_b_policy.player_name)
		
		# Labels and payoffs
		m, n = len(matrix), len(matrix[0])
		if m > 26 or n > 26:
			raise ValueError('Matrix is too large to label with the alphabet')
		alphabet = string.ascii_uppercase
		labels_row, labels_column = [alphabet[i] for i in range(len(matrix))], [alphabet[j] for j in range(len(matrix[0]))]
		payoff_dict = {f'{labels_row[i]},{labels_column[j]}': matrix[i][j] for j in range(n) for i in range(m)}
		
		# Run rounds until game ends
		round_idx = 0
		while round_idx < max_game_rounds:
			if np.random.random() < end_prob:
				break
			else:
				round_outputs = run_round(policy_dict, names_tuple, matrix, labels_row, labels_column, results_dict, end_prob, total_message_rounds)
				results_dict = update_results(results_dict, round_outputs, round_idx, policy_dict, payoff_dict, names_tuple)
				json.dump(results_dict, open(f"{output_dir}/{fname}.json", "w"))
				round_idx += 1

		return results_dict
	except Exception as e:
		logging.exception(e)
		if os.path.exists(f"{output_dir}/{fname}.json"):
			os.remove(f"{output_dir}/{fname}.json")
			logging(f"File {f'{output_dir}/{fname}.json'} deleted due to error.")

def main(config, cli_args):
	# Instantiate Policy directly for both players using parameters from CLI or config
	player_a_policy = Policy(cli_args.model_a or config["player_a_parameters"]["model"],
								cli_args.temperature_a or config["player_a_parameters"]["temperature"],
								cli_args.player_a_name or config["player_a_parameters"]["name"],
								cli_args.a_opponent_name or config["player_b_parameters"]["name"],
								cli_args.player_a_role or config["player_a_parameters"]["role"],
								cli_args.player_a_persona or config["player_a_parameters"]["persona"],
								cli_args.player_a_strategic_reasoning or config["player_a_parameters"]["strategic_reasoning"],
								cli_args.sys_prompt_substitutions_a or config["player_a_parameters"]["sys_prompt_substitutions"],
								cli_args.show_past_reasoning_player_a or config["player_a_parameters"]["show_past_reasoning"],
								cli_args.show_messages_player_a or config["player_a_parameters"]["show_messages"],
								cli_args.show_intended_actions_player_a or config["player_a_parameters"]["show_intended_actions"])

	player_b_policy = Policy(cli_args.model_b or config["player_b_parameters"]["model"],
								cli_args.temperature_b or config["player_b_parameters"]["temperature"],
								cli_args.player_b_name or config["player_b_parameters"]["name"],
								cli_args.b_opponent_name or config["player_a_parameters"]["name"],
								cli_args.player_b_role or config["player_b_parameters"]["role"],
								cli_args.player_b_persona or config["player_b_parameters"]["persona"],
								cli_args.player_b_strategic_reasoning or config["player_b_parameters"]["strategic_reasoning"],
								cli_args.sys_prompt_substitutions_b or config["player_b_parameters"]["sys_prompt_substitutions"],
								cli_args.show_past_reasoning_player_b or config["player_b_parameters"]["show_past_reasoning"],
								cli_args.show_messages_player_b or config["player_b_parameters"]["show_messages"],
								cli_args.show_intended_actions_player_b or config["player_b_parameters"]["show_intended_actions"])


	# Game parameters from CLI or config
	matrix = cli_args.matrix or config["game_parameters"]['game_matrix']
	max_game_rounds = cli_args.max_game_rounds or config["game_parameters"]["max_game_rounds"]
	total_message_rounds = cli_args.total_message_rounds or config["game_parameters"]["total_message_rounds"]
	end_prob = cli_args.end_prob or config["game_parameters"]["end_prob"]
	output_fname = cli_args.output_fname or config["game_parameters"]["output_fname"]

	# Execution parameters from CLI or config
	num_games = cli_args.num_games or config["execution_parameters"]["num_games"]
	max_workers = cli_args.max_workers or config["execution_parameters"]["max_workers"]

	# Parallel processing
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(run_game, player_a_policy, player_b_policy, end_prob=end_prob, fname=f"{output_fname}-{n}", matrix=matrix, max_game_rounds=max_game_rounds, total_message_rounds=total_message_rounds) for n in range(num_games)]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run a game simulation between two players.')

	# Use config file or default to config.json settings (though note that CLI provided args will override config file settings)
	parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')

	# Player A CLI arguments
	parser.add_argument('--model_a', type=str, help='Model for Player A.')
	parser.add_argument('--temperature_a', type=float, help='Temperature for Player A.')
	parser.add_argument('--player_a_name', type=str, help='Name of Player A.')
	parser.add_argument('--player_a_persona', type=str, help='Persona of Player A.')
	parser.add_argument('--a_opponent_name', type=str, help="Name of Player A's opponent")
	parser.add_argument('--player_a_role', type=str, help='Role of Player A.')
	parser.add_argument('--player_a_strategic_reasoning', type=bool, help='Strategic reasoning for Player A.')
	parser.add_argument('--show_past_reasoning_player_a', type=bool, help='Show past reasoning for Player A.')
	parser.add_argument('--show_messages_player_a', type=bool, help='Show messages for Player A.')
	parser.add_argument('--show_intended_actions_player_a', type=bool, help='Show intended actions for Player A.')
	parser.add_argument('--sys_prompt_substitutions_a', type=json.loads, help='System prompt substitutions for Player A.') #Enter JSON string, e.g. '{"key1": "value1", "key2": "value2"}'

	# Player B CLI arguments
	parser.add_argument('--model_b', type=str, help='Model for Player B.')
	parser.add_argument('--temperature_b', type=float, help='Temperature for Player B.')
	parser.add_argument('--player_b_name', type=str, help='Name of Player B.')
	parser.add_argument('--player_b_persona', type=str, help='Persona of Player B.')
	parser.add_argument('--b_opponent_name', type=str, help="Name of Player B's opponent")
	parser.add_argument('--player_b_role', type=str, help='Role of Player B.')
	parser.add_argument('--player_b_strategic_reasoning', type=bool, help='Strategic reasoning for Player B.')
	parser.add_argument('--show_past_reasoning_player_b', type=bool, help='Show past reasoning for Player B.')
	parser.add_argument('--show_messages_player_b', type=bool, help='Show messages for Player B.')
	parser.add_argument('--show_intended_actions_player_b', type=bool, help='Show intended actions for Player B.')
	parser.add_argument('--sys_prompt_substitutions_b', type=json.loads, help='System prompt substitutions for Player B.')

	# Game parameters
	parser.add_argument('--matrix', type=json.loads, help='Game matrix.') #Enter list of lists of lists, e.g. '[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]'
	parser.add_argument('--total_message_rounds', type=int, help='Total message rounds in each game round.')
	parser.add_argument('--num_games', type=int, help='Number of games to run in parallel.')
	parser.add_argument('--end_prob', type=float, help='Game end probability.')
	parser.add_argument('--output_fname', type=str, help='Base filename for output files.')

	# Execution parameters
	parser.add_argument('--max_game_rounds', type=int, help='Maximum game rounds.')
	parser.add_argument('--max_workers', type=int, help='Maximum number of parallel workers.')

	cli_args = parser.parse_args()

	with open(cli_args.config, 'r') as f:
		config = json.load(f)

	main(config, cli_args)