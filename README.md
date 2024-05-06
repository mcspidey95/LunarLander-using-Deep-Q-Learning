![Recording 2024-04-30 101038 (1)](https://github.com/mcspidey95/LunarLander-using-Deep-Q-Learning/assets/90018162/dd4e4268-97cc-4cec-9a87-f4e38e1ecc3f)

**Parameters to Change (Test the Model)**

If you want to test the saved model, change the following parameters before running:
* [Line 126] Edit the Environment to human mode, env = gym.make('LunarLander-v2', render_mode='human')
* [Line 127] Change the n_games value to the number of times you want to test.
* [Line 128] Change epsilon = 0.0 batch_size = 2000 load = True
