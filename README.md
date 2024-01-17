# repeated-games
## Introduction
This script enables LLMs with custom personas to play custom two-player repeated matrix games with each other. Each game consists of one or more rounds. Models are given the opportunity to talk to each other before making a final decision. This is done through one or more "message rounds" within a single round.

## Usage
Make a .env file in the main directory and set environment variables 'OPENAI_API_KEY' and 'OPENAI_ORGANIZATION'.

Experiments are run from the main.py file. Parameters and run_game() feature in main().

First, note that you may specify an mxn matrix with tuple of payoffs for players A and B. For example, a matrix for a Bach or Stravinsky game can be specified as follows:
```python
matrix = [[(10,5), (0, 0)],
	   [(0,0), (5, 10)]]
```
Custom personas can be added to the prompts/personas.json file.
- At the top level, the names of personas are the keys
- Each value at the top level (corresponding to a name-key) is a JSON object that defines the persona.
	- The "intro" begins the system prompt (usually "You are...")
	- "add_context" allows more details about the persona to be added
	- Desired JSON output schema is specified as a JSON object assigned to the "output_schema" value.
		- "reasoning"-, "message"-, and "action"-keys are paired with output instruction string-values
		- Single curly braces surround optional {placeholders} within the instructions

You can set parameters for parallelization of running games, the games themselves, as well as players A and B.
Parallelization parameters: num_games is how many games you'd like to run, max_workers is the maximum number of threads that can be actively executing tasks simultaneously

Game parameters: end_prob is the probability that a game will end prematurely, max_game_rounds specifies the maximum number of rounds per game, and total_message_rounds specifies the number of messages exchanged before a final decision is made by both players.

Filenames of multiple games (run in parallel) follow the format {output_fname}-{n}.json and are found in results/runs-{YYYY}-{MM}-{DD} where the date is the date of running the games.

Player parameters: model (name of model on the OpenAI API), player_name, player_role (the player either determines the 'row' or 'column' of the outcome and is said to have the role of either 'row' or 'column' accordingly), persona (name of persona found in json file), sys_prompt_substitutions allow values to be inserted into placeholders of the system prompt defining the persona; add key-value pairs into the dictionary where the keys are the placeholder names and the values are the corresponding values to insert. Additional strategic_reasoning can be provided to the models; by default this is set to False.

The final three player parameters are show_past_reasoning, show_messages and show_intended_actions. If set to True, game history of previous rounds presented to instances of LLMs will contain such elements.

## Credits
Jesse Clifton & Tim Chan
