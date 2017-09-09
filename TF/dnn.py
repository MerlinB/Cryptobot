import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Deep:

    def __init__(self, train_data, hl1=500,hl2=500,hl3=500,n_classes=10,batch_size=10, input_size=784, hm_epochs=10):
        self.n_nodes_hl1 = hl1
        self.n_nodes_hl2 = hl2
        self.n_nodes_hl3 = hl3
        self.n_classes = n_classes
        self.batch_size = batch_size
        
        self.input_size = input_size
        self.hm_epochs = hm_epochs
        
        
        self.train_data = train_data
        # self.train_labels = train_labels
        # self.test_data = test_data
        # self.test_labels = test_labels
        self.x = tf.placeholder('float', [None, self.input_size]) # rows will change with batch number.
        self.y = tf.placeholder('float')


    def neural_network_model(self,data):
        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([self.input_size, self.n_nodes_hl1])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl2]))}

        hidden_3_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
                          'biases':tf.Variable(tf.random_normal([self.n_nodes_hl3]))}

        output_layer = {'weights':tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
                        'biases':tf.Variable(tf.random_normal([self.n_classes])),}


        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

        return output

    def train_neural_network(self, x):
        prediction = self.neural_network_model(x)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y) )
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        
        with tf.Session() as sess:
            
            # Necessary to initiate variables.
            sess.run(tf.global_variables_initializer())
        
            for epoch in range(self.hm_epochs):
                epoch_loss = 0
                
                # Walks through datachunks of batch size at a time, so RAM doesn't run empty
                # for i in range(int(self.train_data/self.batch_size)):
                #     epoch_x = self.train_data[i*self.batch_size:(i+1)*self.batch_size]
                #     print(i,epoch_x)
                #     epoch_y = self.train_data
                dataset = self.train_data
                for date in dataset:
                    epoch_x = [float(dataset[date]['hash-rate']), 
                                float(dataset[date]['median-confirmation-time']), 
                                float(dataset[date]['mempool-count']),
                                float(dataset[date]['mempool-size']),
                                float(dataset[date]['miners-revenue']),
                                float(dataset[date]['n-transactions-excluding-popular']),
                                float(dataset[date]['trade_volume'])]
                    epoch_y = [float(dataset[date]['market-price'])]
                    # returns optimizer, cost. feed_dict needs to set placeholder.
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, self.y: epoch_y})
                    epoch_loss += c
        
                print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print('done')
            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            # 
            # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # print('Accuracy:',accuracy.eval({x:self.test_data, self.y:self.test_labels}))
            
    def start(self):
        self.train_neural_network(self.x)

    
    
