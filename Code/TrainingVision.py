import os
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
    
from Models.VisionModel import CVAE
import time
 

def ds_gen():
    dirname = 'C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsCarRacing'
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

if __name__ == "__main__": 
    model_save_path = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/VaeCarRacing"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)
    shuffle_size = 20 * 1000 
    ds = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(100)
    ds = ds.prefetch(100) # prefetch 100 batches in the buffer #tf.data.experimental.AUTOTUNE)
    
    vae = CVAE()
    tensorboard_callback.set_model(vae)
    loss_weights = [1.0, 1.0] # weight both the reconstruction and KL loss the same
    losses = {'reconstruction': vae.get_loss().get('reconstruction'), 'KL':vae.get_loss().get('KL')}
    vae.compile(optimizer=vae.optimizer, loss=losses, loss_weights=loss_weights) 
    step = 0

    losses_per_epoch_rec = []
    losses_per_epoch_kl = []
    time_taken = []
    filename_rec = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"rec"+"lossCarRacing"+".npy"
    filename_kl = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"kl"+"lossCarRacing"+".npy"
    filename_time = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"timeTakenWithTransfer"+".npy"

    start = time.time()

    for i in range(10):
        j = 0
        for x_batch in ds:
            if i == 0 and j == 0:
                vae._set_inputs(x_batch)
            loss = vae.train_on_batch(x=x_batch, y=x_batch, return_dict=True) 
            j += 1
            step += 1 

            [tf.summary.scalar(loss_key, loss_val, step=step) for loss_key, loss_val in loss.items()] 
            if j % 100 == 0:
                index = 0
                output_log = 'epoch: {} mb: {}'.format(i, j)
                for loss_key, loss_val in loss.items():
                    output_log += ', {}: {:.4f}'.format(loss_key, loss_val)
                    if index == 1:
                        losses_per_epoch_kl.append(loss_val)
                    elif index==2:
                        losses_per_epoch_rec.append(loss_val)
                    index+=1
                print(output_log)
        print('saving')
        end = time.time()
        time_taken.append(end-start)
        print(end - start)
        print(start)
        print(end)
        np.save(filename_time, np.asarray(time_taken))
        tf.keras.models.save_model(vae, model_save_path, include_optimizer=True, save_format='tf')

    np.save(filename_rec, np.asarray(losses_per_epoch_rec))
    np.save(filename_kl, np.asarray(losses_per_epoch_kl))
