import gym
import minerl
import logging

# define environment
logging.basicConfig(level=logging.DEBUG) #DEBUG, INFO, WARNING
env = gym.make('MineRLNavigateDense-v0')
obs = env.reset()

# start taking actions
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)