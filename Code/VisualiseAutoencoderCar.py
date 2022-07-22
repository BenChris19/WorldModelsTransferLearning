import numpy as np
import os
import tensorflow as tf
import random
from Models.VisionModel import CVAE
import matplotlib.pyplot as plt

#Testing

DATA_DIR = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/rolloutsCarRacing"
model_path_name = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_vaeTensorflowLast"
other_model = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_vaeTensorflowLastTransfered"
other_model2 = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/data_vaeTensorflowLastTransferedPretrained" 

filelist = os.listdir(DATA_DIR)

obs = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["obs"]
obs = obs.astype(np.float32)/255.0
frame = obs[600].reshape(1, 64, 64,3)
plt.axis('off')
plt.imshow(frame[0])
plt.show()

vae = CVAE()

vae.set_weights(tf.keras.models.load_model(model_path_name, compile=False).get_weights())

batch_z = vae.encode(frame)
print(batch_z[0]) # print out sampled z
reconstruct = vae.decode(batch_z)
plt.axis('off')
plt.imshow(reconstruct[0])
plt.show()

vae = CVAE()

vae.set_weights(tf.keras.models.load_model(other_model, compile=False).get_weights())

batch_z = vae.encode(frame)
print(batch_z[0]) # print out sampled z
plt.axis('off')
reconstruct = vae.decode(batch_z)
plt.imshow(reconstruct[0])
plt.show()

vae = CVAE()

vae.set_weights(tf.keras.models.load_model(other_model2, compile=False).get_weights())

batch_z = vae.encode(frame)
print(batch_z[0]) # print out sampled z
plt.axis('off')
reconstruct = vae.decode(batch_z)
plt.imshow(reconstruct[0])
plt.show()