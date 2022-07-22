import numpy as np
import random
import os

from env import make_env
from Models.ControllerModel import make_controller

dir_name = 'C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsCarRacing'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

controller = make_controller()
total_frames = 0
env = make_env(gym_env = "CarRacing-v0", dream_env=False, full_episode=False, load_model=False)

for trial in range(10000):

    random_generated_int = random.randint(0, 2**31-1)
    filename = dir_name+"/"+str(random_generated_int)+".npz"
    recording_frame = []
    recording_action = []
    recording_reward = []
    recording_done = []

    np.random.seed(random_generated_int)
    env.seed(random_generated_int)
 
    controller.init_random_model_params(stdev=np.random.rand()*0.01)
    
    tot_r = 0
    frame, obs = env._reset() # pixels

    for i in range(1000): 
      env.render() 

      recording_frame.append(frame)
      action = controller.get_action(obs) 

      recording_action.append(action)

      frame, obs, reward, done, info = env._step(action) #_step

      tot_r += reward

      recording_reward.append(reward)
      recording_done.append(done)

      if done:
        print('total reward {}'.format(tot_r))
        break

    total_frames += (i+1)
    print('total reward {}'.format(tot_r))
    print("dead at", i+1, "total recorded frames for this worker", total_frames)

    recording_frame = np.array(recording_frame, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    recording_done = np.array(recording_done, dtype=np.bool)
    
    if (len(recording_frame) > 100):
      np.savez_compressed(filename, obs=recording_frame, action=recording_action, reward=recording_reward, done=recording_done)
      env.close()
      env = make_env(gym_env = "CarRacing-v0", dream_env=False, full_episode=False, load_model=False)
env.close()