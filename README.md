# repeated-games
## Introduction
This script enables LLMs with custom personas to play custom two-player repeated matrix games with each other. Each game consists of one or more rounds. Models are given the opportunity to talk to each other before making a final decision. This is done through one or more "message rounds" within a single round.

## Usage
Make a .env file in the main directory and set environment variables 'OPENAI_API_KEY', 'OPENAI_ORGANIZATION', and 'ANTHROPIC_API_KEY'.

You may specify an mxn matrix with tuple of payoffs for players A and B. For example, a matrix for a Bach or Stravinsky game can be specified as follows:
```python
matrix = [[(10,5), (0, 0)],
	   [(0,0), (5, 10)]]
```

Player A and B will play a game for max_game_rounds rounds unless the game prematurely ends with end_prob probability at each round.

Experiments are run from the main.py file. You can either provide experiment parameters in a config file (e.g. config.json) or input them through the command-line interface. For help run: `python main.py -help`

Descriptions of parameters:
{
    "game_parameters": {
        "end_prob": <float: the probability of ending the game at rounds before max_game_rounds>,
        "max_game_rounds": <int: the maximum number of rounds of the game; this will be the number of rounds per game unless the game prematurely ends with end_prob probability at each round>,
        "total_message_rounds": <int: the number of messages exchanged per round>>,
        "output_fname": <str:  the base name of the output file>,
        "game_matrix": <list: a list of lists of tuples containing the rewards for each player; the reward for >
    },
    "player_a_parameters": {
        "model": <str: the model used for setting up player Aj>,
        "temperature": <float: the termperature the model uses for player A>,
        "name": <str: the name given to player A>,
        "role": <str: the role of the player is either row or colimn (default row)>,
        "persona": <str: the persona of player A>,
        "strategic_reasoning": <bool: whether the player will use strategic reasoning examples in their prompt>,
        "show_past_reasoning": <bool: whether the player will see their past reasoning when sending messages and making decisions>,
        "show_messages": <bool: whether the player will see past messages when sending messages and making decisions>,
        "show_intended_actions": <bool: whether the player will see their intended actions when sending messages and making decisions>,
        "sys_prompt_substitutions": <dict: match desired values to keys in order to replace possible placeholders (which have the names of keys) from the system prompt>
    },
    "player_b_parameters": {
        "model": <str: the model used for setting up player Bj>,
        "temperature": <float: the termperature the model uses for player B>,
        "name": <str: the name given to player B>,
        "role": <str: the role of the player is either row or colimn (default column)>,
        "persona": <str: the persona of player B>,
        "strategic_reasoning": <bool: whether the player will use strategic reasoning examples in their prompt>,
        "show_past_reasoning": <bool: whether the player will see their past reasoning when sending messages and making decisions>,
        "show_messages": <bool: whether the player will see past messages when sending messages and making decisions>,
        "show_intended_actions": <bool: whether the player will see their intended actions when sending messages and making decisions>,
        "sys_prompt_substitutions": <dict: match desired values to keys in order to replace possible placeholders (which have the names of keys) from the system prompt>
    },
    "execution_parameters": {
        "num_games": <int: the number of games to run; games will be run in parallel if possible>,
        "max_workers": <int: the maximum number of threads that can be actively executing tasks simultaneously>
    }
}

Custom personas can be added to the prompts/personas.json file.
- At the top level, the names of personas are the keys
- Each value at the top level (corresponding to a name-key) is a JSON object that defines the persona.
	- The "intro" begins the system prompt (usually "You are...")
	- "add_context" allows more details about the persona to be added
	- Desired JSON output schema is specified as a JSON object assigned to the "output_schema" value.
		- "reasoning"-, "message"-, and "action"-keys are paired with output instruction string-values
		- Single curly braces surround optional {placeholders} within the instructions

Filenames of multiple games (run in parallel) follow the format {output_fname}-{n}.json and are found in results/runs-{YYYY}-{MM}-{DD} where the date is the date of running the games.

## Credits
Jesse Clifton & Timothy Chan.