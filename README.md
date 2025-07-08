***Git clone the repository***

***Open it in pycharm and wait for venv to be created automatically***

***Create .env file and paste GOOGLE_API_KEY=<your-api-key>***

***If not using pycharm try this but I havent tested it:***

```
uv venv
```
***Activate the environment venv then***
```
uv pip install -r pyproject.toml 
```
The prompts for the agents are in the .prompts file