# repeated-games
## Introduction
This script enables LLMs with custom personas to play two-player repeated matrix games with each other. Each game consists of one or more rounds. Models are given the opportunity to talk to each other before making a final decision. This is done through one or more "message rounds" within a single round.

## Usage
Make a .env file in the main directory and set environment variables 'OPENAI_API_KEY' and 'OPENAI_ORGANIZATION'.

Experiments are run from the main.py file. 

First, note that you may specify an mxn matrix with tuple of payoffs for players A and B. For example, a matrix for a Bach or Stravinsky game can be specified as follows:
	matrix = [[(10,5), (0, 0)],
 [(0,0), (5, 10)]]

Custom personas can be added to the prompts/personas.csv file.

You can set parameters for the game, as well as players A and B.

Game parameters: end_prob is the probability that a game will end prematurely, max_game_rounds specifies the maximum number of rounds per game, and total_message_rounds specifies the number of messages exchanged before a final decision is made by both players.

Filenames of multiple games (run in parallel) follow the format {output_fname}-{n}.json and are found in results/runs-{YYYY}-{MM}-{DD} where the date is the date of running the games.

Player parameters: model (name of model on the OpenAI API), player_name, player_role (the player either determines the 'row' or 'column' of the outcome and is said to have the role of either 'row' or 'column' accordingly), persona (name of persona found in CSV file), sys_prompt_substitutions allow values to be inserted into placeholders of the system prompt defining the persona; add key-value pairs into the dictionary where the keys are the placeholder names and the values are the corresponding values to insert. Additional strategic_reasoning can be provided to the models; by default this is set to False.

The final three player parameters are show_past_reasoning, show_messages and show_intended_actions. If set to True, game history presented to instances of LLMs will contain such elements.

