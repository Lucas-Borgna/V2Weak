from keras import backend as K
import numpy as np
import tensorflow as tf

global bag_size
bag_size = 200


def test_loss(ytrue, ypred):
	loss = -K.sum(ypred)/bag_size - K.sum(ytrue)/bag_size
	loss = K.square(loss)
	return loss 


def symmetric_square_loss(ytrue, ypred):
	yq = K.sum(ytrue)/bag_size
	yg = 1 - yq
	fq = K.sum(ypred)/bag_size
	fg = 1 - fq
	loss = (fq - yq) * (fq - yq) + (fg - yg) * (fg - yg)
	return loss

def ksquare_loss(ytrue,ypred):
	return K.mean(K.square(ypred - ytrue), axis=-1)
	
 
def square_loss(ytrue, ypred):
	loss = K.sum(ypred)/bag_size - K.sum(ytrue)/bag_size
	loss = K.square(loss)
	#np.save('helpme.npy',ypred)
	return loss 

def cross_entropy_loss(ytrue, ypred):
	yq = K.sum(ytrue)/batch_size
	yg = 1-yq

	fq = K.sum(ypred)/batch_size	
	fg = 1-fq
	loss = -tf.multiply(yq,K.log(K.abs(fq/yq)+0.0001)) - tf.multiply(yg,K.log(K.abs(fg/yg)+0.0001))
	return loss 

def loss_function(ytrue, ypred):
    #Assuming that ypred contains the same ratio replicated
    loss = K.sum(ypred)/ypred.shape[0] - K.sum(ytrue)/ypred.shape[0]
    loss = K.square(loss)
    return loss


