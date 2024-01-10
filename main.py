import os
from copy import deepcopy
import logging
import json
import string
from collections import defaultdict
import numpy as np
import pandas as pd
import openai
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_ORGANIZATION')

"""
Function calling hierarchy:
main() calls run_game() multiple times in parallel and initializes objects of class Policy
run_game() calls run_round() and update_results()
run_round() calls query_policy() and results_to_history_str()
results_to_history_str() goes into query_policy()
query_policy() calls get_response_openai()
get_response_openai() uses get_prompts()
get_prompts() uses matrix_to_str()
"""

class Policy:
	def __init__(self, model, player_name, opponent_name, role, persona, strategic_reasoning=False, sys_prompt_substitutions={}, show_past_reasoning=True, show_intended_actions=True):
		self.player_name = player_name 
		self.opponent_name = opponent_name # for prompt construction
		self.role = role # for determining whether the player acts on rows or columns of matrix
		self.persona = persona # for selecting data from csv file
		self.strategic_reasoning = strategic_reasoning # whether to include optional strategic reasoning
		self.sys_prompt_substitutions = sys_prompt_substitutions # for inserting more expressions into sys prompt (values replace keys in curly braces within template str)
		self.model = model
		self.show_past_reasoning = show_past_reasoning
		self.show_intended_actions = show_intended_actions
		self.most_recent_action = None

	def query_policy(self, history_str, message_round, labels_row, labels_column, end_prob, matrix, total_message_rounds, **kwargs):
		output = get_response_openai(self.player_name, self.opponent_name, self.role, self.persona, end_prob, history_str, matrix, 
										labels_row, labels_column, message_round, total_message_rounds, strategic_reasoning=self.strategic_reasoning,
										sys_prompt_substitutions = self.sys_prompt_substitutions, model=self.model, persona=self.persona, **kwargs)
		self.most_recent_action = output['action']
		return output

def always_defect(player_name, opponent_name, end_prob, results):
	answer = '{"reasoning": "", "action": "D"}'
	return answer

def get_strategic_reasoning_prompt(examples_to_include=('simultaneous_22')):
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

	# import prompt/personas.csv and get the intro, add_context, and output_schema values for the persona
	prompts_df = pd.read_csv('./prompts/personas.csv')
	prompts_df = prompts_df[prompts_df['persona'] == persona]
	intro = prompts_df['intro'].iloc[0]
	add_context = prompts_df['add_context'].iloc[0]
	output_schema = prompts_df['output_schema'].iloc[0]

	system = ""
	system += f"{intro} " if intro else ""
	system += f"Your name is {player_name}, and the agent you are playing against is named {opponent_name}. At each time-step, you and {opponent_name} will exchange messages. {total_message_rounds} messages in total will be exchanged, with the first speaker being chosen at random at each time-step. At each time-step, you and {opponent_name} will simultaneously choose your actions; the available actions are labeled {','.join(labels)}. The game has a {end_prob}\% chance of ending after every time-step. Here is the payoff matrix:\n{matrix_str}.\n\nBefore each time-step, you will be shown a summary of the history of play thus far."
	system += f" {add_context} " if add_context else ""
	system += f"You should return a JSON with the following format (key names should be in double quotes; you must always fill out every field):\n\n{output_schema}"
	# Include any values that have not been added
	system = system.format_map(defaultdict(str, **sys_prompt_substitutions))

	# Construct user prompt
	if message_round > total_message_rounds:
		final_message_round = "There are no more message rounds."
	elif message_round in (total_message_rounds-1, total_message_rounds):
		final_message_round = f"It is message round {message_round} of {total_message_rounds}. This is your last chance to send a message."
	else:
		final_message_round = f"It is message_round {message_round} of {total_message_rounds}."
	user = f"Here is a summary of the history of play thus far, in the order ({row}, {column}): {history_str}.\n\n{final_message_round} Now please output a correctly-formatted JSON:"
	return system, user

# Sends request and gets JSON response from the OpenAI API. In the future, we may want to add get_response_anthropic etc.
def get_response_openai(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round=None, total_message_rounds=None, strategic_reasoning=False, sys_prompt_substitutions={}, model='gpt-4-32k', **kwargs):
	system, user = get_prompts(player_name, opponent_name, role, persona, end_prob, history_str, matrix, labels_row, labels_column, message_round, total_message_rounds, strategic_reasoning=strategic_reasoning, sys_prompt_substitutions=sys_prompt_substitutions)
	max_retries = 5
	retry_interval_sec = 20
	answer_dict = {}
	for _ in range(max_retries):
		try:
			completion = openai.ChatCompletion.create(model=model, messages=[{'role': 'system', 'content': system, }, {'role': 'user', 'content': user}], max_tokens=800, temperature=1.0, **kwargs)
			# Not valid JSON if there are too many tokens and it gets cut off
			if completion.choices[0]['finish_reason'] != "stop":
				raise Exception("Completion error: finish_reason is not stop")
			answer = completion['choices'][0]['message']['content']
			answer_dict = json.loads(answer)
			break
		except (
			openai.error.RateLimitError,
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
		#print('Failed to get response from OpenAI API')
		logging.error('Failed to get response from OpenAI API')
 
	return answer_dict

def results_to_history_str(policy, names_tuple, results, round_outputs=None, messages=False):
	history_str = ''
	player_a, player_b = names_tuple[0], names_tuple[1]
	show_past_reasoning = policy.show_past_reasoning
	show_intended_actions = policy.show_intended_actions
	current_player = policy.player_name
	# Summarize previous rounds using results
	for i, round_ in results['rounds']:
		history_str += f'Round {i+1}:\n'
		# Summarize any combination of {reasoning, messages, actions}
		for j, message_round in round_['message_rounds']:
			if show_past_reasoning and (message_round[j]['player'] == current_player):
				history_str += f"Your reasoning: {message_round[j]['reasoning']}\n"
			
			if messages and (message_round[j]['player'] == current_player):
				history_str += f"Your message: {message_round[j]['message']}\n"

			if messages and (message_round[j]['player'] != current_player):
				history_str += f"Other party's message: {message_round[j]['message']}\n"

			if show_intended_actions and (message_round[j]['player'] == current_player):
				history_str += f"Your intended action at t={j+1} within this round: {message_round[j]['action']}\n"

			history_str += "\n"
		
		# Summarize final actions and round outcomes
		if i+1 == len(round_):
			your_payoff = round_['outcome'][-1][f'{player_a}_payoff'] if current_player == player_a else round_['outcome'][-1][f'{player_b}_payoff']
			your_final_action = round_['message_round'][-1]['action'] if round_['message_round'][-1]['player'] == current_player else round_['message_round'][-2]['action']

			other_payoff = round_['outcome'][-1][f'{player_a}_payoff'] if current_player != player_a else round_['outcome'][-1][f'{player_b}_payoff']
			other_final_action = round_['message_round'][-1]['action'] if round_['message_round'][-1]['player'] != current_player else round_['message_round'][-2]['action']
   
			history_str += f"Outcome of round {i+1}: At the end of the round, you chose {your_final_action}. On the other hand, the other party chose {other_final_action}. This resulted in a payoff of {your_payoff} for you and a payoff of {other_payoff} for the other party.\n"
		
		history_str += "\n"
	
	# Summarize the new round so far, if there is info
	if round_outputs:
		history_str += f"Current round:\n"
		for j, message_round in round_outputs:
			player = message_round[0]
			reasoning = message_round[1]['reasoning']
			message = message_round[1]['message']
			action = message_round[1]['action']
			if show_past_reasoning and (player == current_player):
				history_str += f"Your reasoning: {reasoning}\n"
			
			if messages and (player == current_player):
				history_str += f"Your message: {message}\n"

			if messages and (player != current_player):
				history_str += f"Other party's message: {message}\n"

			if show_intended_actions and (player == current_player):
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
		output = policy.query_policy(history_str, ix+1, labels_row, labels_column, end_prob, matrix, total_message_rounds) # LLM thinks and speaks here
		round_outputs.append((speaker, output))
		
	# In the case where no messages are exchanged, both players only select actions (to avoid messages, message_round is made > total_message_rounds).
	if total_message_rounds == 0:
		speaker_index = first_speaker_index
		speaker, non_speaker = names_tuple[speaker_index], names_tuple[1-speaker_index]
		policy = policy_dict[speaker]
		history_str = results_to_history_str(policy, names_tuple, results_dict, round_outputs)
		output = policy.query_policy(history_str, total_message_rounds+1, labels_row, labels_column, end_prob, matrix, total_message_rounds) # LLM thinks and speaks here
		round_outputs.append((speaker, output))
 
	# Get final action from the player who didn't get to send the final message (if there are message rounds) and the second player (if there are no message rounds)
	last_index = 1 - speaker_index
	speaker, non_speaker = names_tuple[last_index], names_tuple[1-last_index]
	policy = policy_dict[speaker]
	history_str = results_to_history_str(policy, names_tuple, results_dict, round_outputs)
	output = policy.query_policy(history_str, total_message_rounds + 1, labels_row, labels_column, end_prob, matrix, total_message_rounds)
	round_outputs.append((speaker, output))

	return round_outputs

def update_results(results_dict, round_outputs, i, policy_dict, payoff_dict, names_tuple):
	"""
	Update summary statistics for game using latest round. Example JSON below.
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
	results_dict['summary'][f'{player_a}_avg_utility'] = results_dict['summary'][f'{player_a}_total_utility'] / (i+1)
	results_dict['summary'][f'{player_b}_avg_utility'] = results_dict['summary'][f'{player_b}_total_utility'] / (i+1)
	
	results_dict['rounds'].append({})
	results_dict['rounds'][-1]['message_rounds'] = []
	results_dict['rounds'][-1]['outcome'] = {f'{player_a}_payoff': player_a_payoff, f'{player_b}_payoff': player_b_payoff}
	for k, message_round in enumerate(round_outputs):
		player = message_round[0]
		reasoning = message_round[1]['reasoning']
		message = message_round[1]['message']
		action = message_round[1]['action']
		results_dict['rounds'][-1]['message_rounds'].append({'player': player, 'reasoning': reasoning, 'message': message, 'action': action})

	return results_dict

def run_game(player_a_policy, player_b_policy, end_prob: float, fname: str, matrix: list[list], max_game_rounds: int = 20, total_message_rounds: int = 3):
	# Check errors and if passed, make deep copies of policies
	if (not isinstance(player_a_policy, Policy)) or (not isinstance(player_b_policy, Policy)):
		raise TypeError('Argument is not an instance of the Policy class')
	if player_a_policy.player_name == player_b_policy.player_name:
		raise ValueError('Players cannot have the same name')
	if fname == "":
		raise ValueError('Filename cannot be empty')
	if not (0 <= end_prob <= 1):
		raise ValueError('Probability of termination out of range')

	player_a_policy = deepcopy(player_a_policy)
	player_b_policy = deepcopy(player_b_policy)
	 
	# Initialize values and dictionaries to store results and values
	results_dict = {'summary': {}, 'rounds': []}
	results_dict['summary']['player_a_total_utility'] = 0.
	results_dict['summary']['player_b_total_utility'] = 0.
	policy_dict = {player_a_policy.player_name: player_a_policy, 
					player_a_policy.player_name: player_b_policy}
	names_tuple = (player_a_policy.player_name, player_b_policy.player_name)

	# Check if runs-{today_date} directory exists, if not, create it
	today_date = time.strftime("%Y-%m-%d")
	output_dir = f"results/runs-{today_date}"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# Labels and payoffs
	m, n = len(matrix), len(matrix[0])
	if m > 26 or n > 26:
		raise ValueError('Matrix is too large to label with the alphabet')
	alphabet = string.ascii_uppercase
	labels_row, labels_column = [alphabet[i] for i in range(len(matrix))], [alphabet[j] for j in range(len(matrix[0]))]
	payoff_dict = {f'{labels_row[i]},{labels_column[j]}': matrix[i][j] for j in range(n) for i in range(m)}
	
	# Run rounds until game ends
	round_num = 0
	while round_num < max_game_rounds:
		if np.random.random() < end_prob:
			break
		else:
			round_outputs = run_round(policy_dict, names_tuple, matrix, labels_row, labels_column, results_dict, end_prob, total_message_rounds)
			results_dict = update_results(results_dict, round_outputs, round_num, policy_dict, payoff_dict, names_tuple)
			json.dump(results_dict, open(f"{output_dir}/{fname}.json", "w"))
			round_num += 1
	return results_dict

def main():	
	### Matrix for Bach or Stravinsky
	matrix = [[(10,5), (0, 0)],
			[(0,0), (5, 10)]]
	### Modify matrix to suit needs
	
	# Game Parameters
	end_prob = 0.001
	num_games = 5
	max_game_rounds = 16
	total_message_rounds = 3 # One or more or zero "message_rounds" in each round in a game
	output_fname = f"vanilla-symmetric-bos-16-turns"

	# Player A Parameters
	model_a = 'gpt-4'
	player_a_name = "Alice"
	player_a_role = "row"
	player_a_persona = 'vanilla'
	sys_prompt_substitutions_player_a = {}
	player_a_strategic_reasoning = True
	show_past_reasoning_player_a = True
	show_intended_actions_player_a = True
	
	# Player B Parameters
	model_b = 'gpt-4'
	player_b_name = "Bob"
	player_b_role = "column"
	player_b_persona = 'vanilla'
	sys_prompt_substitutions_player_b = {}
	player_b_strategic_reasoning = True
	show_past_reasoning_player_b = True
	show_intended_actions_player_b = True
	
	# Initialize policies. Deepcopies are made in run_game()
	player_a_policy = Policy(model_a, player_a_name, player_b_name, player_a_role, player_a_persona, player_a_strategic_reasoning, sys_prompt_substitutions_player_a, show_past_reasoning_player_a, show_intended_actions_player_a)
	player_b_policy = Policy(model_b, player_b_name, player_a_name, player_b_role, player_b_persona, player_b_strategic_reasoning, sys_prompt_substitutions_player_b, show_past_reasoning_player_b, show_intended_actions_player_b)
	
	# Run games in parallel
	max_workers = 4
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(run_game, player_a_policy, player_b_policy, end_prob = end_prob, fname=f"{output_fname}-{n}", matrix=matrix, max_game_rounds=max_game_rounds, total_message_rounds = total_message_rounds) for n in range(num_games)]

if __name__ == "__main__":
	main()