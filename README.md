![Recording 2024-04-30 101038 (1)](https://github.com/mcspidey95/LunarLander-using-Deep-Q-Learning/assets/90018162/dd4e4268-97cc-4cec-9a87-f4e38e1ecc3f)

**Parameters to Change (Test the Model)**

If you want to test the saved model, change the following parameters before running:
* [Line 126] Edit the Environment to human mode, <kbd>env = gym.make('LunarLander-v2', render_mode='human')</kbd>
* [Line 127] Change the <kbd>n_games</kbd> value to the number of times you want to test.
* [Line 128] Change <kbd>epsilon = 0.0</kbd> <kbd>batch_size = 2000</kbd> <kbd>load = True</kbd>



# Environment Documentation

## Description
This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

## Action Space
There are four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

## Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Rewards
Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points. If the lander moves away from the landing pad, it loses reward. If the lander crashes, it receives an additional -100 points. If it comes to rest, it receives an additional +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

## Starting State
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

## Episode Termination
The episode finishes if:

- the lander crashes (the lander body gets in contact with the moon);

- the lander gets outside of the viewport (x coordinate is greater than 1);

- the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:
