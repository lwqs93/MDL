import matplotlib.pyplot as plt
import numpy as np
import keras
#-----------------------create dataset for input--------------------------------------
def create_dataset_input(data,look_back,look_ahead):
    dataX=[]
    for i in range(len(data)-look_back-look_ahead):
        c=[]
        for j in range(look_back):
            a = data[i+j, 0:36]
            a= np.reshape(a,(4,9))
            a= np.array([a]).T
            c.append(a)
        dataX.append(c)
    return np.array(dataX)
#---------create dataset with the input shape----------------------------------------
def create_dataset_output(data,look_back,look_ahead):
    dataY=[]
    for i in range(len(data)-look_back-look_ahead):
        b=data[(i + look_back):(i + look_back+look_ahead), 0:36]
        b = np.reshape(b, (36*look_ahead))
        dataY.append(b)
    return np.array(dataY)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()