import numpy as np
import os
import tensorflow as tf
from Models.TransferVisionModel import OwnVae

DATA_DIR = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsLunar" 
SERIES_DIR = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/processed_dataLunarLander" 
model_path_name = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/VisionLunar" 
if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def ds_gen():
    filenames = os.listdir(DATA_DIR)[:10000] # only use first 10k episodes
    for _, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(DATA_DIR, fname)
        if os.stat(file_path).st_size > 0:
          data = np.load(file_path)
          img = data['obs']
          action = np.reshape(data['action'], newshape=[-1, 2]) #2
          reward = data['reward']
          done = data['done']

        
        n_pad = 1000 - img.shape[0] # pad so they are all a thousand step long episodes
        img = tf.pad(img, [[0, n_pad], [0, 0], [0, 0], [0, 0]])
        n_pad = 1000 - action.shape[0]
        action = tf.pad(action, [[0, n_pad], [0, 0]])
        n_pad = 1000 - reward.shape[0]
        reward = tf.pad(reward, [[0, n_pad]])
        n_pad = 1000 - done.shape[0]
        done = tf.pad(done, [[0, n_pad]], constant_values=done[-1])

        yield img, action, reward, done

def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32, tf.float32, tf.bool),
    output_shapes=((1000, 64, 64, 3), (1000, 2), (1000,), (1000,))) 
    return dataset

@tf.function
def encode_batch(batch_img):
  simple_obs = batch_img/255.0
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  return mu, logvar

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = create_tf_dataset()
dataset = dataset.batch(1, drop_remainder=True)

vae = OwnVae()

vae.set_weights(tf.keras.models.load_model(model_path_name, compile=False).get_weights())
mu_dataset = []
logvar_dataset = []
action_dataset = []
r_dataset = []
d_dataset = []

i=0
for batch in dataset:
  i += 1
  obs_batch, action_batch, r, d = batch
  obs_batch = tf.squeeze(obs_batch, axis=0)
  action_batch = tf.squeeze(action_batch, axis=0)
  r = tf.reshape(r, [-1, 1])
  d = tf.reshape(d, [-1, 1])

  mu, logvar = encode_batch(obs_batch)

  mu_dataset.append(mu.numpy().astype(np.float16))
  logvar_dataset.append(logvar.numpy().astype(np.float16))
  action_dataset.append(action_batch.numpy())
  r_dataset.append(r.numpy().astype(np.float16))
  d_dataset.append(d.numpy().astype(np.bool))

  if ((i+1) % 100 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)
r_dataset = np.array(r_dataset)
d_dataset = np.array(d_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "seriesLunar.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset, reward=r_dataset, done=d_dataset)