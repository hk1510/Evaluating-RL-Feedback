from gymnasium.wrappers.record_video import RecordVideo
from parking_agent_env_new import ParkingAgentEnv # Custom version of parking environment
from stable_baselines3 import SAC
from pygame.locals import *
from HighwayEnv.highway_env.envs.parking_env import ParkingEnv
from video_overlay import show_video
import sys
import csv

FEEDBACK = int(sys.argv[1]) # 0: control grp 1, 1: Test grp 1, 2: control grp 2, 3: test grp 2
NAME = sys.argv[2]
TEST_EPS = 3
TRAIN_EPS = 5
seed_list = [27, 8, 19, 7, 6, 478, 2, 8, 77, 20]
test_seed_list = [0, 4200, 856]

og_env = ParkingEnv()

model = SAC.load("SAC_2e5.zip", env=og_env) # trained decision-making agent

if FEEDBACK == 1:
    feedback = 2 # 0 = no feedback, 1 = verification, 2 = correct response, 3 = error flagging
else:
    feedback = 0

def create_human_env():
    env = ParkingAgentEnv(feedback=feedback, model=model, render_mode='rgb_array') # this is the custom environment we edited.
    env.config["manual_control"] = True
    env.config["screen_height"] = 720
    env.config["screen_width"] = 1280
    env.config['vehicles_count'] = 0
    env.config['real_time_rendering'] = True
    env.config['simulation_frequency'] = 15 # default 15
    if FEEDBACK == 3:
        return RecordVideo(env, video_folder='./recordings', name_prefix='human', episode_trigger=lambda x: True)
    return env


def create_agent_env():
    env = ParkingEnv(render_mode="rgb_array")
    env.config["screen_height"] = 720
    env.config["screen_width"] = 1280
    env.config['vehicles_count'] = 0
    env.config['real_time_rendering'] = True
    env.config['simulation_frequency'] = 15
    return RecordVideo(env, video_folder='./recordings', name_prefix='agent', episode_trigger=lambda x: True)

def create_test_env():
    env = ParkingAgentEnv(feedback=0, model=model, render_mode="rgb_array")
    env.config["manual_control"] = True
    env.config["screen_height"] = 720
    env.config["screen_width"] = 1280
    env.config['vehicles_count'] = 0
    env.config['real_time_rendering'] = True
    env.config['simulation_frequency'] = 15
    return env

def play_ep(env, seed):
    done = truncated = False
    env.reset(seed=seed)
    ret = 0
    while not done:
        # env controlled by arrow keys instead
        obs, reward, done, truncated, info = env.step(env.action_space.sample()) 
        ret += reward
        env.render()
        if done or truncated:
            env.close()
            break
    env.close()
    return ret, info


#env.reset() # NOTE: Need to 'x' out the window once for screen res update
human_env = create_human_env()  
test_env = create_test_env()
if FEEDBACK == 2 or FEEDBACK == 3: 
    agent_env = create_agent_env()

returns = []
for i in range(TEST_EPS): # Practice the exercise 3 times
    ret, info = play_ep(test_env, test_seed_list[i])
    returns.append(ret)
    returns.append(info)


for i in range(TRAIN_EPS): # Practice the exercise 3 times
    # play_ep(human_env, seed_list[i])
    done = truncated = False
    obs, info = human_env.reset(seed=seed_list[i])
    ret = 0
    while not done:
        # env controlled by arrow keys instead
        obs, reward, done, truncated, info = human_env.step(human_env.action_space.sample()) 
        ret += reward
        human_env.render()
        if done or truncated:
            human_env.close()
            break
    
    human_env.close()
    returns.append(ret)
    returns.append(info)

    # Agent plays if test control or test grp 2
    if FEEDBACK == 2 or FEEDBACK == 3:
        done = truncated = False
        obs, info = agent_env.reset(seed=seed_list[i])
        while not done:
            act, _ = model.predict(obs)
            # env controlled by agent
            obs, reward, done, truncated, info = agent_env.step(act)
            agent_env.render()
            if done or truncated:
                agent_env.close()
                break
        agent_env.close()
        if FEEDBACK == 3:
            show_video(i)
    
for i in range(TEST_EPS): # Practice the exercise 3 times
    ret, info = play_ep(test_env, test_seed_list[i])
    returns.append(ret)
    returns.append(info)

returns.append(FEEDBACK)
returns.append(NAME)
with open('results_new_seeds.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow(returns)

