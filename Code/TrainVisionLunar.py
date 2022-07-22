import tensorflow as tf
import os
import numpy as np
from Models.TransferVisionModel import OwnVae
import time

def ds_gen():
    dirname = 'C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsLunar'
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    for _, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        if os.stat(file_path).st_size > 0:
            data = np.load(file_path)
            for _, img in enumerate(data['obs']):
              img_i = img / 255.0
              yield img_i

model_save_path = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/VisionLunar"
tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
summary_writer = tf.summary.create_file_writer(tensorboard_dir)
summary_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)


ds = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
ds = ds.shuffle(20 * 1000, reshuffle_each_iteration=True).batch(100)
ds = ds.prefetch(100) 


ownVae = OwnVae()
tensorboard_callback.set_model(ownVae)
loss_weights = [1.0, 1.0] # weight both the reconstruction and KL loss the same
losses = {'reconstruction': ownVae.get_loss().get('reconstruction'), 'KL':ownVae.get_loss().get('KL')}
ownVae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses, loss_weights=loss_weights) #ownVae.get_loss()
step = 0

losses_per_epoch_rec = []
losses_per_epoch_kl = []
time_taken = []
filename_rec = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"rec"+"lossLunarLander"+".npy"
filename_kl = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"kl"+"lossLunarLander"+".npy"
filename_time = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"timeTakenWithTransferLunar"+".npy"

start = time.time()

for i in range(10):
    j = 0
    for x_batch in ds:
        if i == 0 and j == 0:
            ownVae._set_inputs(x_batch)

        loss = ownVae.train_on_batch(x=x_batch, y=x_batch, return_dict=True) 
        j += 1
        step += 1 
        [tf.summary.scalar(loss_key, loss_val, step=step) for loss_key, loss_val in loss.items()] 
        if j % 100 == 0: 
            output_log = 'epoch: {} mb: {}'.format(i, j)
            for loss_key, loss_val in loss.items():
                output_log += ', {}: {:.4f}'.format(loss_key, loss_val)
            print(output_log)
    print('saving')
    end = time.time()
    time_taken.append(end-start)
    print(end - start)
    print(start)
    print(end)
    tf.keras.models.save_model(ownVae, model_save_path, include_optimizer=True, save_format='tf') 
