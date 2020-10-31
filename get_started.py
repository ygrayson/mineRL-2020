import gym
import minerl
import logging

# define environment
logging.basicConfig(level=logging.INFO) #DEBUG, INFO, WARNING
env = gym.make('MineRLNavigateDense-v0')
obs = env.reset()

# random agent
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)

# compass angle agent
done = False
net_reward = 0

while not done:
    action = env.action_space.noop()

    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(action)

    net_reward += reward
    #print(action)
    #print("Reward: ", reward)
    print("Total reward: ", net_reward)
