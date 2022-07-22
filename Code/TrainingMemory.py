import numpy as np
import os
import json
import tensorflow as tf
import time

from Models.MemoryModel import MDNRNN, sample_vae
    
DATA_DIR = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/processed_dataCarRacing"
model_save_path = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_rnn_car"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
data_r = raw_data["reward"]
data_d = raw_data["done"]
N_data = len(data_mu) # should be 10k

# save 1000 initial mu and logvars. Used for sampling when training in dreams
initial_z_save_path = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_rnn/tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

initial_mu = []
initial_logvar = []
for i in range(1000): #1000
  mu = np.copy(data_mu[i][0, :]*10000).astype(np.int).tolist()
  logvar = np.copy(data_logvar[i][0, :]*10000).astype(np.int).tolist()
  initial_mu.append(mu)
  initial_logvar.append(logvar)
with open(os.path.join(initial_z_save_path, "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

def ds_gen():
  for _ in range(4000):
    indices = np.random.permutation(N_data)[0:100] #
    # suboptimal b/c we are always only taking first set of steps
    mu = data_mu[indices][:, :1000] 
    logvar = data_logvar[indices][:, :1000]
    action = data_action[indices][:, :1000]
    z = sample_vae(mu, logvar)
    r = tf.cast(data_r[indices], tf.float16)[:, :1000]
    d = tf.cast(data_d[indices], tf.float16)[:, :1000]
    yield z, action, r, d
    
dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float16, tf.float16, tf.float16, tf.float16), \
    output_shapes=((100,1000,32), \
    (100, 1000, 3), \
    (100, 1000, 1), \
    (100, 1000, 1)))
dataset = dataset.prefetch(10)
tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
summary_writer = tf.summary.create_file_writer(tensorboard_dir)
summary_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)

rnn = MDNRNN()
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss())
tensorboard_callback.set_model(rnn)

# train loop:

losses_per_epoch_mem = []
filename_mem = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"mem"+"lossMemoryFreezeFC"+".npy"

start = time.time()
step = 0
for raw_z, raw_a, raw_r, raw_d in dataset:
    curr_learning_rate = (0.001-0.00001) * (1.0) ** step + 0.00001
    rnn.optimizer.learning_rate = curr_learning_rate
    
    inputs = tf.concat([raw_z, raw_a], axis=2)

    if step == 0:
        rnn._set_inputs(inputs)

    dummy_zero = tf.zeros([raw_z.shape[0], 1, raw_z.shape[2]], dtype=tf.float16)
    z_targ = tf.concat([raw_z[:, 1:, :], dummy_zero], axis=1) # zero pad the end but we don't actually use it
    z_mask = 1.0 - raw_d
    z_targ = tf.concat([z_targ, z_mask], axis=2) # use a signal to not pass grad

    outputs = {'MDN': z_targ}
    loss = rnn.train_on_batch(x=inputs, y=outputs, return_dict=True)
    [tf.summary.scalar(loss_key, loss_val, step=step) for loss_key, loss_val in loss.items()]

    if (step%100==0 and step > 0):
        end = time.time()
        time_taken = end-start
        start = time.time()
        output_log = "step: %d, train_time_taken: %.4f, lr: %.6f" % (step, time_taken, curr_learning_rate)
        for loss_key, loss_val in loss.items():
            output_log += ', {}: {:.4f}'.format(loss_key, loss_val)
            losses_per_epoch_mem.append(loss_val)
        print(output_log)
    if (step%1000==0 and step > 0):
        tf.keras.models.save_model(rnn, model_save_path, include_optimizer=True, save_format='tf')
    step += 1

np.save(filename_mem, np.asarray(losses_per_epoch_mem))