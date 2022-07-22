import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


from Models.MemoryModel import MDNRNN

rnn = MDNRNN()
@tf.function
def sample_vae(vae_mu, vae_logvar):
    sz = vae_mu.shape[1]
    mu_logvar = tf.concat([vae_mu, vae_logvar], axis=1)
    z = tfp.layers.DistributionLambda(lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :sz], scale_diag=tf.exp(theta[:, sz:])), dtype=tf.float16)
    return z(mu_logvar)

class ownMDNRNN(tf.keras.Model):
    def __init__(self):
        super(ownMDNRNN, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

        
        self.inference_base =  tf.keras.layers.LSTM(units=256, return_sequences=True, return_state=True, time_major=False) #rnn.inference_base

        self.rnn_size = 256
        self.rnn_num_mixture = 5
        self.z_size = 32
        self.rnn_d_true_weight = 1.0
        self.rnn_batch_size = 100
        self.rnn_max_seq_len = 1000
        self.rnn_input_seq_width = 34 

        rnn_out_size = 5 * 32 * 3 
        self.out_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.rnn_size),
            tf.keras.layers.Dense(rnn_out_size, name="mu_logstd_logmix_net")])

        super(ownMDNRNN, self).build((self.rnn_batch_size, self.rnn_max_seq_len, self.rnn_input_seq_width))

    def get_loss(self):
        num_mixture = self.rnn_num_mixture
        batch_size = self.rnn_batch_size
        z_size = self.z_size
        d_true_weight = self.rnn_d_true_weight
        
        """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
        # Construct a loss function with the right number of mixtures and outputs
        def z_loss_func(y_true, y_pred):
            '''
            This loss function is defined for N*k components each containing a gaussian of 1 feature
            '''
            mdnrnn_params = y_pred
            y_true = tf.reshape(y_true, [batch_size, -1, z_size + 1]) # +1 for mask
            z_true, mask = y_true[:, :, :-1], y_true[:, :, -1:]
            # Reshape inputs in case this is used in a TimeDistribued layer
            mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3*num_mixture], name='reshape_ypreds')
            vae_z, mask = tf.reshape(z_true, [-1, 1]), tf.reshape(mask, [-1, 1])
            
            out_mu, out_logstd, out_logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1, name='mdn_coef_split')
            out_logpi = out_logpi - tf.reduce_logsumexp(input_tensor=out_logpi, axis=1, keepdims=True) # normalize
            logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
            lognormal = -0.5 * ((vae_z - out_mu) / tf.exp(out_logstd)) ** 2 - out_logstd - logSqrtTwoPI
            v = out_logpi + lognormal
            
            z_loss = -tf.reduce_logsumexp(input_tensor=v, axis=1, keepdims=True)
            mask = tf.reshape(tf.tile(mask, [1, z_size]), [-1, 1]) # tile b/c we consider z_loss is flattene
            z_loss = mask * z_loss # don't train if episode ends
            z_loss = tf.reduce_sum(z_loss) / tf.reduce_sum(mask) 
            return z_loss
        losses = {'MDN': z_loss_func}

        return losses

    def parse_rnn_out(self, out):
        mdnrnn_param_width = self.rnn_num_mixture * self.z_size * 3 # 3 comes from pi, mu, std 
        mdnrnn_params = out[:, :mdnrnn_param_width]

        r = None
        d_logits = None

        return mdnrnn_params, r, d_logits
    def call(self, inputs, training=True):
        return self.__call__(inputs, training)
    def __call__(self, inputs, training=True):
        rnn_out, _, _ = self.inference_base(inputs, training=training)
        rnn_out = tf.reshape(rnn_out, [-1, self.rnn_size])
        out = self.out_net(rnn_out)
        mdnrnn_params, r, d_logits = self.parse_rnn_out(out)
       
        outputs = {'MDN': mdnrnn_params} # can't output None b/c tfkeras redirrects to loss for optimization 

        return outputs