
# coding: utf-8

# In[53]:

import tensorflow as tf
from dataset import TextDataset

import numpy as np


# In[5]:

data = TextDataset('books/book_EN_democracy_in_the_US.txt')


# In[103]:

tf.reset_default_graph()

x = tf.placeholder(shape=(None, None), dtype=tf.int32)
y = tf.placeholder(shape=(None, None), dtype=tf.int32)

x_oh = tf.one_hot(indices=x, depth=data.vocab_size)
y_oh = tf.one_hot(indices=y, depth=data.vocab_size)

outputs0, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(num_units=128, initializer=tf.orthogonal_initializer),
                               dtype=tf.float32,
                               inputs=x_oh, scope='LSTM0')

outputs1, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(num_units=128, initializer=tf.orthogonal_initializer),
                               dtype=tf.float32,
                               inputs=outputs0, scope='LSTM1')

logits = tf.layers.dense(outputs1, data.vocab_size)
loss = tf.losses.softmax_cross_entropy(y_oh, logits)

train = tf.train.AdamOptimizer(0.002).minimize(loss)


# In[104]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[105]:

for i in range(10000):
    x_, y_ = data.batch(32,30)
    sess.run(train, {x: x_, y: y_})
    
    if i % 100 == 0:
        x_, y_ = data.batch(32,30)
        print(i, sess.run(loss, {x: x_, y: y_}))
        
        x_, _ = data.batch(5,1)
        for i in range(30):
            x_ = np.hstack((x_, np.argmax(sess.run(logits, {x: x_})[:,-1:,:], -1)))

        print('\n'.join([data.convert_to_string(i) for i in x_]))


# In[78]:




# In[ ]:



