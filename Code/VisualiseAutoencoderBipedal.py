from Models.TransferVisionModel import OwnVae
import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt

#Testing

filelist = os.listdir("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsBipedal")
obs = np.load(os.path.join("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsBipedal", random.choice(filelist)))["obs"]
obs = obs.astype(np.float32)/255.0
frame = obs[60].reshape(1, 64, 64,3)
plt.axis('off')
plt.imshow(frame[0])
plt.show()

ownvae = OwnVae()
ownvae.set_weights(tf.keras.models.load_model("C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/tryVAEBIpedal",compile=False).get_weights())

batch_z = ownvae.encode(frame)
reconstruct = ownvae.decode(batch_z)
plt.axis('off')
plt.imshow(reconstruct[0])
plt.show()
