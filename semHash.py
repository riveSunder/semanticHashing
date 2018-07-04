
import numpy as np
import time

import tensorflow as tf


import matplotlib
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

# User defined flags
#tf.app.flags.DEFINE_boolean('restore', False,"""Restore previously trained model""")
tf.app.flags.DEFINE_integer('maxSteps', 100,"""number of epochs""")
tf.app.flags.DEFINE_integer('dispIt', 20,"""display every nth iteration""")
tf.app.flags.DEFINE_integer('batchSize', 45,"""Number of entries per minibatch""")
tf.app.flags.DEFINE_float('lR', 3e-7,"""learning rate""")
tf.app.flags.DEFINE_boolean('dispFigs', False,"""Whether to save figures""")
tf.app.flags.DEFINE_float('dORate', 0.125,"""rate for applying dropout""")



maxSteps = FLAGS.maxSteps
dispIt = FLAGS.dispIt
batchSize = FLAGS.batchSize
lR = FLAGS.lR
dispFigs = FLAGS.dispFigs
dORate = FLAGS.dORate

if(0):
    myData = np.load('./data/XBio.npy')
    
    #myData = myData.T
elif(0):
    myData = np.load('./data/myRaman.npy')
elif(1):
    myData = np.load('./data/openaiVecJobDescriptions.npy')

print(myData.shape,np.mean(myData),np.min(myData),np.max(myData))

# Each word frequency entry is a feature, so we should probably normalize them all to be similar?
#for cj in range(len(myData)):
#    myData[cj,:] = myData[cj,:] / (np.max(myData[cj,:])+1e-2)

#myData = myData.T # examples in first dimension

np.random.seed(42)
myVal = myData[0:5,:]
myTrain = myData[5:len(myData)-1,:]

lenData = (myData.shape)[1]

data = tf.placeholder("float",[None,lenData], name='data')


h1Size = 1024
h2Size = 512
codeSize = 128
#display dimensions for encodings
dimX= 16  
dimY = 8

def semHash(data):
    """# A fully-connected autoencoder for semantic hashing vector representations of documents"""

    fc1 = tf.layers.dense(data, h1Size, activation=tf.nn.sigmoid)
    do1 = tf.nn.dropout(fc1,(1-dORate))

    fc2 = tf.layers.dense(do1, h2Size, activation=tf.nn.sigmoid)
    do2 = tf.nn.dropout(fc2,(1-dORate))

    fc3 = tf.layers.dense(do2, h2Size, activation=tf.nn.sigmoid)
    do3 = tf.nn.dropout(fc3,(1-dORate))

    codes = tf.layers.dense(do3, codeSize, activation=tf.nn.sigmoid)
    do4 = tf.nn.dropout(codes,(1-dORate))

    fc5 = tf.layers.dense(do4, h2Size, activation=tf.nn.sigmoid)
    do5 = tf.nn.dropout(fc5,(1-dORate))

    fc6 = tf.layers.dense(do5, h2Size, activation=tf.nn.sigmoid)
    do6 = tf.nn.dropout(fc6,(1-dORate))

   
    fc7 = tf.layers.dense(do6, h1Size, activation=tf.nn.sigmoid)
    do7 = tf.nn.dropout(fc7,(1-dORate))
    
    rebow = tf.layers.dense(do7,lenData)
    
    #print layer sizes for debugging
    print("fc1 size: ",fc1.shape)    
    print("fc2 size: ",fc2.shape)
    print("fc3 size: ",fc3.shape)
    print("fc5 size: ",fc5.shape)
    print("fc6 size: ",fc6.shape)
    print("fc7 size: ",fc7.shape)
    
    print("encoding size: ",codes.shape)
    print("decoded size: ",rebow.shape)
    

    return rebow, codes 

rebow, codes = semHash(data)
print("output shape rebow and codes: ", rebow.shape,codes.shape)
# objective function and training

# MSE
loss = tf.losses.mean_squared_error(data,rebow)

#loss = ((tf.pow(rebow-data, 2)))

trainOp = tf.train.AdamOptimizer(
	learning_rate=lR,beta1=0.9,
	beta2 = 0.999,
	epsilon=1e-08,
	use_locking=False,
	name='Adam').minimize(loss,global_step
=tf.train.get_global_step())


mySaver = tf.train.Saver()

init = tf.global_variables_initializer()

def main(unused_argv):
    t0 = time.time()
    with tf.Session() as sess:
        sess.run(init)

        for ck in range(maxSteps):

            inp = tf.placeholder(tf.float32)
            myMean = tf.reduce_mean(inp)
            np.random.shuffle(myTrain)
            for cm in range(0,batchSize,len(myTrain)):
                input_ = myTrain[ck:ck+batchSize,:]
                #print(input_.shape)
                [batchLoss, trainOp_]  = sess.run([loss, trainOp], feed_dict = {data: input_})
                #print("batch training loss: %.3e"%myMean.eval(feed_dict={inp:batchLoss}))


            if (ck%dispIt == 0):
                #report training and validation losses
                #mySaver.save(sess,'./models/'+myModel,global_step=i)\

                # Get training and validation loss scores
                input_ = myTrain
                inVal_ = myVal

                myTemp = (sess.run(loss, feed_dict={data: input_}))
                myLossTrain = myMean.eval(feed_dict={inp: myTemp})

                myTemp = (sess.run(loss, feed_dict={data: inVal_}))
                myLossVal = myMean.eval(feed_dict={inp: myTemp})
                elapsed = time.time()-t0
                print("Epoch %i %.2f s elapsed, training loss, validation loss: %.4e, %.4e"%(ck,elapsed,myLossTrain,myLossVal))

                [rebowT, codesT] = sess.run([rebow,codes],feed_dict = {data: input_})
                [rebowV, codesV] = sess.run([rebow,codes],feed_dict = {data: inVal_})

                meanCodes = tf.reduce_mean(codesT,axis=0)
                if(1):#dispFigs):
                    # Display figures of progress
                    plt.figure(figsize=(20,20))
                    cl = 10
                    plt.subplot(3,3,1)
                    plt.plot(input_[cl,:])
                    plt.title("input Bag o' Words",fontsize=18)
                    plt.ylabel("training",fontsize=32)
                    plt.subplot(3,3,2)
                    plt.imshow(np.reshape(codesT[cl,:],(dimX,dimY)))
                    plt.title("encoding",fontsize=18)
                    plt.subplot(3,3,3)
                    plt.plot(rebowT[cl,:])
                    plt.title("decoded Bag o' Words",fontsize=18)
                    
                    cl = 20
                    plt.subplot(3,3,4)
                    plt.plot(input_[cl,:])
                    plt.title("input Bag o' Words",fontsize=18)
                    plt.ylabel("training",fontsize=32)
                    plt.subplot(3,3,5)
                    plt.imshow(np.reshape(codesT[cl,:],(dimX,dimY)))
                    plt.title("encoding",fontsize=18)
                    plt.subplot(3,3,6)
                    plt.plot(rebowT[cl,:])
                    plt.title("decoded Bag o' Words",fontsize=18)
                    
                    cl = 3
                    plt.subplot(3,3,7)
                    plt.plot(inVal_[cl,:])
                    plt.title("input Bag o' Words",fontsize=18)
                    plt.ylabel("validation",fontsize=32)
                    plt.subplot(3,3,8)
                    plt.imshow(np.reshape(codesV[cl,:],(dimX,dimY)))
                    plt.title("encoding",fontsize=18)
                    plt.subplot(3,3,9)
                    plt.plot(rebowV[cl,:])
                    plt.title("decoded Bag o' Words",fontsize=18)
                    plt.savefig('./figs/trainingSemHash%i.png'%ck)
        # save the resulting codes

        np.save('./output/valCodes.npy',codesV)
        np.save('./output/trainCodes.npy',codesT)

if __name__ == "__main__":
    tf.app.run()
