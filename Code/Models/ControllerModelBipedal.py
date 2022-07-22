import numpy as np
import random
import json


def make_controller():
  controller = Controller()
  return controller

def clip(x, lo=-1.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class Controller:
  ''' simple one layer model for Bipedal Walker '''
  def __init__(self):
    self.env_name = 'BipedalWalker-V2'
    self.exp_mode = 4 
    self.input_size = 32 + 1 * 256
    self.z_size = 32
    self.a_width = 4 

    self.weight = np.random.randn(self.input_size, self.a_width)
    self.bias = np.random.randn(self.a_width)
    self.param_count = (self.input_size)*self.a_width+self.a_width

    self.render_mode = False

  def get_action(self, h):
    action = np.tanh(np.dot(h, self.weight) + self.bias) 
    return action

  def set_model_params(self, model_params):
    self.bias = np.array(model_params[:self.a_width])
    self.weight = np.array(model_params[self.a_width:]).reshape(self.input_size, self.a_width)


  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)

def simulateBipedal(controller, env,  render_mode=True, num_episode=5, seed=-1):
  reward_list = []
  t_list = []


  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)


  for episode in range(num_episode):
    print('episode: {}/{}'.format(episode, num_episode))

    obs, frame = env._reset()
    total_reward = 0.0
    for t in range(1000):
      env.render()

      action = controller.get_action(obs) 
      frame, obs, reward, done, info = env._step(action, frame) 
      total_reward += reward
      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)
  return reward_list, t_list

  