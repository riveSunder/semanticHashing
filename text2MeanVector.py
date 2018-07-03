"""This is a wrapper for tranforming text (in this case job descriptions) using 
OpenAI's model from 'Unsupervised Sentiment Neuron' a char-level LSTM model
	Source info: 
	Model: https://github.com/openai/generating-reviews-discovering-sentiment
	Paper: https://arxiv.org/pdf/1704.01444.pdf
	Blog: https://blog.openai.com/unsupervised-sentiment-neuron/
"""

import numpy as np
from encoder import Model
model = Model()
import time

myJDs = np.load('../data/myJDs.npy')

print(myJDs.shape)
myDim = 4096

t0 = time.time()
for ck in range(len(myJDs)):
    t1 = time.time()
    tempX = model.transform(myJDs[ck])
    print(tempX.shape)
    tempX = np.mean(tempX,axis=0)
    if(ck == 0):
        X = np.reshape(tempX,(1,myDim))
    else:
        X = np.append(X,np.reshape(tempX,(1,myDim)),axis=0)
    elapsed = time.time()-t1

    print('job description number %i parsed in %.3f s'%(ck,elapsed))

np.save('../data/openaiVecJobDescriptions.npy',X)
elapsed = time.time()-t0
print('finished after %.3f s'%elapsed)
