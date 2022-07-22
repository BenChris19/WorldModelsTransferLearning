import numpy as np
import tensorflow as tf

from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.box2d.bipedal_walker import BipedalWalker
from LunarLander import LunarLander


class CarRacingWrapper(CarRacing):
  def __init__(self, full_episode=False):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def _process_frame(self, frame):
    obs = frame[0:84, :, :]
    obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self).step(action)
    return self._process_frame(obs), reward, done, {}

  def _reset(self):
    return self._process_frame(super(CarRacingWrapper, self).reset())

class LunarLanderWrapper(LunarLander):
  def __init__(self, full_episode=False):
    super(LunarLanderWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def _process_frame(self, frame):
    obs = Image.fromarray(frame, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _step(self, action, frame):
    _, reward, done, _ = super(LunarLanderWrapper, self).step(action)
    return self._process_frame(frame), reward, done, {}

  def _reset(self):
    super(LunarLanderWrapper, self).reset()
    frame = super(LunarLanderWrapper, self).render("rgb_array")
    return self._process_frame(frame)

class BipedalWalkerWrapper(BipedalWalker):
  def __init__(self, full_episode=False):
    super(BipedalWalkerWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def _process_frame(self, frame):
    obs = Image.fromarray(frame, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _step(self, action, frame):
    _, reward, done, _ = super(BipedalWalkerWrapper, self).step(action)
    return self._process_frame(frame), reward, done, {}

  def _reset(self):
    super(BipedalWalkerWrapper, self).reset()
    frame = super(BipedalWalkerWrapper, self).render("rgb_array")
    return self._process_frame(frame)  


from Models.VisionModel import CVAE
from Models.MemoryModel import MDNRNN, rnn_next_state, rnn_init_state

from Models.TransferVisionModel import OwnVae
from Models.MemoryModelLunar import ownMDNRNN

from Models.MemoryModelBipedal import BipedalMDNRNN

class CarRacingMDNRNN(CarRacingWrapper):
  def __init__(self, load_model=True, full_episode=False):
    super(CarRacingMDNRNN, self).__init__(full_episode=full_episode)
    self.vae = CVAE()
    self.rnn = MDNRNN()
     
    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/VaeCarRacing").variables])
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_rnn_car").variables])

    self.rnn_states = rnn_init_state(self.rnn)
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    z = self.vae.encode(result)
    return z

  def _reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    obs = super(CarRacingMDNRNN, self)._reset() # calls step
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])

    z_state = tf.concat([z, h], axis=-1)
    return obs, z_state

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingMDNRNN, self)._step(action)
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])

    z_state = tf.concat([z, h], axis=-1)
    if action is not None:
      self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    return obs, z_state, reward, done, {}

  def close(self):
    super(CarRacingMDNRNN, self).close()
    tf.keras.backend.clear_session()

class LunarLanderMDNRNN(LunarLanderWrapper):
  def __init__(self, load_model=True, full_episode=False):
    super(LunarLanderMDNRNN, self).__init__(full_episode=full_episode)
    self.vae = OwnVae()
    self.rnn = ownMDNRNN()
     
    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/VisionLunar").variables]) 
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_rnn_lunar").variables])

    self.rnn_states = rnn_init_state(self.rnn)
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    z = self.vae.encode(result)
    return z

  def _reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    obs = super(LunarLanderMDNRNN, self)._reset() # calls step
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])

    z_state = tf.concat([z, h], axis=-1)
    return z_state, obs

  def _step(self, action, frame):
    obs, reward, done, _ = super(LunarLanderMDNRNN, self)._step(action,frame)
    z = tf.squeeze(self.encode_obs(frame))
    h = tf.squeeze(self.rnn_states[0])

    z_state = tf.concat([z, h], axis=-1)
    #action = np.append(action,[0])
    self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    return obs, z_state, reward, done, {}

  def close(self):
    super(LunarLanderMDNRNN, self).close()
    tf.keras.backend.clear_session()    

class BipedalWalkerMDNRNN(BipedalWalkerWrapper):
  def __init__(self, load_model=True, full_episode=False):
    super(BipedalWalkerMDNRNN, self).__init__(full_episode=full_episode)
    self.vae = OwnVae()
    self.rnn = BipedalMDNRNN()
     
    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/VisionBipedal").variables]) 
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_rnn_bipedal").variables]) 

    self.rnn_states = rnn_init_state(self.rnn)
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    z = self.vae.encode(result)
    return z

  def _reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    obs = super(BipedalWalkerMDNRNN, self)._reset() # calls step
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])

    z_state = tf.concat([z, h], axis=-1)
    return z_state, obs

  def _step(self, action, frame):
    obs, reward, done, _ = super(BipedalWalkerMDNRNN, self)._step(action,frame)
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])

    z_state = tf.concat([z, h], axis=-1)
    self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    return obs, z_state, reward, done, {}

  def close(self):
    super(BipedalWalkerMDNRNN, self).close()
    tf.keras.backend.clear_session()    


def make_env(gym_env, dream_env=False, full_episode=False, load_model=False):
  if dream_env:
    raise ValueError('training in dreams for carracing is not yet supported')

  elif gym_env == "LunarLander-v2":
    print("Making real LunarLander environment")
    env = LunarLanderMDNRNN(full_episode=full_episode, load_model=load_model)

  elif gym_env == "CarRacing-v0":
    print('making real CarRacing environment')
    env = CarRacingMDNRNN(full_episode=full_episode, load_model=load_model)

  elif gym_env == "BipedalWalker-v2":
    print('making real BipedalWalker environment')
    env = BipedalWalkerMDNRNN(full_episode=full_episode, load_model=load_model)
  return env