from keras.models import Model
from keras.layers import Input, Masking, LSTM, Dense, RepeatVector, TimeDistributed, Activation
from keras import backend as K
import numpy as np
from numpy.random import seed as random_seed
random_seed(123)

max_sentence_length = 5
character_number = 3 # valid character 'a, b' and placeholder '#'

X = np.array([[[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
              [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]])
y_true = np.array([[[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]], # the batch is ['##abb','#babb'], padding '#'
              [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]])

input_tensor = Input(shape=(max_sentence_length, character_number))
masked_input = Masking(mask_value=0)(input_tensor)
encoder_output = LSTM(10, return_sequences=False)(masked_input)
repeat_output = RepeatVector(max_sentence_length)(encoder_output)
decoder_output = LSTM(10, return_sequences=True)(repeat_output)
output = Dense(3, activation='softmax')(decoder_output)

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.all(K.equal(K.variable(y_true), mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy

mask_value = np.array([0, 0, 1])
masked_categorical_crossentropy = get_loss(mask_value)
model = Model(input_tensor, output)
model.compile(loss=masked_categorical_crossentropy, optimizer='adam')
model.summary()



y_pred = model.predict(X)
print('y_pred:', y_pred)
print('y_true:', y_true)
print('model.evaluate:', model.evaluate(X, y_true))
# See if the loss computed by model.evaluate() is equal to the masked loss
import tensorflow as tf
logits=tf.constant(y_pred, dtype=tf.float32)
target=tf.constant(y_true, dtype=tf.float32)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(logits),axis=2))
losses = -tf.reduce_sum(target * tf.log(logits),axis=2)
sequence_lengths=tf.constant([3,4])
mask = tf.reverse(tf.sequence_mask(sequence_lengths,maxlen=max_sentence_length),[0,1])
losses = tf.boolean_mask(losses, mask)
masked_loss = tf.reduce_mean(losses)
with tf.Session() as sess:
    c_e = sess.run(cross_entropy)
    m_c_e=sess.run(masked_loss)
    print("tf unmasked_loss:", c_e)
    print("tf masked_loss:", m_c_e)