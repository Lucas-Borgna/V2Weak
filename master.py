from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import time
import h5py
import pydot
import matplotlib.pyplot as plt
import keras
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping
import tensorflow as tf
import pydot
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

import networks.FC     as FC
import networks.LeNet5 as LeNet5
import networks.VGG19  as VGG19
import networks.VGG16  as VGG16
import networks.resnet as resnet
import networks.bn_shallow as BN_LeNet5

import utilities.datahandler as datahandler
import utilities.analysis    as analysis
from   utilities.loss_function import square_loss as square_loss
from   utilities.loss_function import cross_entropy_loss as cross_entropy_loss
from   utilities.loss_function import symmetric_square_loss as symmetric_square_loss
from   utilities.loss_function import bag_size
from utilities.datahandler import sliceanddice as sliceanddice

from datapath import samples_path, fractions_path, sig_test, bkg_test
from random import shuffle


def data_generator(samples, output):
	num_batches = len(samples)
        z = zip(samples,output)
        shuffle(z)
        samples, output = zip(*z)
	while 1:
	    for i in xrange(num_batches):
		yield samples[i],output[i]



def load_data():

    X  = np.load(samples_path)
    Y  = np.load(fractions_path)


    return X, Y


def create_model(network, input_shape, img_channels, img_rows, img_cols, nb_classes):
    print("Acquring Network Model: ")
    if network == "LeNet5":
        model = LeNet5.GetNetArchitecture(input_shape)
        model_name = "LeNet5"
    elif network == "VGG16":
        model = VGG16.GetNetArchitecture(input_shape)
        model_name = "VGG16"
    elif network == "VGG19":
        model = VGG19.GetNetArchitecture(input_shape)
        model_name = "VGG19"

    elif network == "resnet18":
        model = resnet.ResnetBuilder.build_resnet18((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet18"
    elif network == "resnet34":
        model = resnet.ResnetBuilder.build_resnet18((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet34"

    elif network == "resnet50":
        model = resnet.ResnetBuilder.build_resnet18((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet50"

    elif network == "resnet101":
        model = resnet.ResnetBuilder.build_resnet18((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet101"

    elif network == "resnet152":
        model = resnet.ResnetBuilder.build_resnet18((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet152"

    elif network =="FC":
        model = FC.GetNetArchitecture(input_shape)
        model_name = "Fully Connected"

    elif network == 'BN_LeNet5':
        model = BN_LeNet5.GetNetArchitecture(input_shape)
        model_name = 'BN_LeNet5'

    return model, model_name

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, data_augmentation):

    if data_augmentation == 'False':
        print ('Not Using Data Augmentation.')
        history = model.fit_generator(data_generator(X_train, Y_train),
                                      nb_steps_per_epoch, 
                                      epochs=nb_epoch,
                                      validation_data=data_generator(X_test, Y_test),
                                      nb_val_samples=valsize,
                                      verbose=1)

        return history
    elif data_augmentation == 'True':
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images  # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=(X_test, Y_test),
                              epochs=nb_epoch, verbose=1, max_q_size=100)
        return history


if __name__ == "__main__":

    np.random.seed(1337)
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S")

    batch_size = 500
    nb_classes = 2
    nb_epoch = 30
    img_channels, img_rows, img_cols = 1, 25, 25
    test_size = 0.7
    backend = K.image_dim_ordering()
    k_folds = 1
    Nbags = 10
    nb_steps_per_epoch = 50
    valsize = Nbags
    Ebags = bag_size

    data_augmentation = 'False'
    savemodel = 'True'
    save_schematic ='False'
    save_plots = 'True'
    save_roc = 'True'
    save_cm = 'True'
    precision_recall = 'True'

    scriptname = os.path.basename(__file__)


    network = "BN_LeNet5"

    loss_function = 'square_loss'
    optimizer = 'adadelta'

    SamplesArray, FractionsArray = load_data()

    for j in range (0, k_folds):
        if k_folds == 1:
            X_train, X_test, Y_train, Y_test, input_shape, trainsize, valsize = sliceanddice(SamplesArray, FractionsArray, test_size, Nbags, Ebags, backend, img_rows, img_cols)
        else:
            X_train, X_test, Y_train, Y_test, input_shape = datahandler.create_fold(sigArray, bkgArray, j, k_folds, backend, img_channels, img_rows, img_cols)

        print ("Running Fold: ",j+1 ,"/", k_folds)

        model = None # Clearing the NN.
        model, model_name = create_model(network, input_shape, img_channels, img_cols, img_rows, nb_classes)
        model.compile(loss = square_loss, optimizer = optimizer, metrics = ['accuracy'])

        history = train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, data_augmentation)


    model_storage = "/mnt/storage/lborgna/TrainedModels/TrainWeak/"
    info_storage = model_storage +"info/"

    mylist1 = ['SamplesArray: ', 'FractionsArray: ', 'Sig Test', 'Sig Test',"Time and Date: ",'Backend: ', 'input_shape', "Network Model: ", "Epochs: ", 'batch_size: ', 'test_size: ',  " K folds: ", "Data Augmentation: ","Loss Function: ", "Optimizer: ", 'Bag size', 'Number of bags', 'nb_steps_per_epoch']
    mylist2 = [samples_path, fractions_path, sig_test,bkg_test,timestamp, backend, input_shape ,model_name, nb_epoch, batch_size,test_size, k_folds, data_augmentation, loss_function, optimizer, bag_size, Nbags, nb_steps_per_epoch]

    redundancy = "/home/lborgna/NN/V2Weak/info/"

    if savemodel:
        print ("Storing Information File: ")
        datahandler.writeinfofile(mylist1 = mylist1, mylist2 = mylist2, folder = info_storage, scriptname = scriptname, timestamp = timestamp)
        datahandler.writeinfofile(mylist1 = mylist1, mylist2 = mylist2, folder = redundancy, scriptname = scriptname, timestamp = timestamp)
        print("Saving Model: Final")
        namestr = os.path.basename(__file__)
        ext = ".h5"
        model.save(model_storage + namestr + model_name + timestamp + ext)

    #if save_schematic:
    #    schematic_name = model_name+timestamp+"schematic.png"
    #    plot_model("/home/lborgna/NN/V2Weak/schematics/"+schematic_name)
    #    print("schematic saved: ", schematic_name)

    if save_plots:
        fig, ax1 = plt.subplots()
        plt.grid(b=True, which = 'major', color='k', linestyle = '-')
        plt.grid(b=True, which = 'minor', color='k', linestyle = '-')
        ax2 = ax1.twinx()
        acc, = ax1.plot(history.history['acc'], 'g-')
        valacc, = ax1.plot(history.history['val_acc'], 'g--')

        loss, = ax2.plot(history.history['loss'], 'b-')
        valloss, = ax2.plot(history.history['val_loss'], 'b--')
        ax1.set_xlabel('Number of Epochs')
        ax1.set_ylabel('Accuracy', color = 'g')
        ax1.tick_params(axis='y', colors='green', which='both')
        ax1.yaxis.label.set_color('g')
        ax2.set_ylabel('loss', color = 'b')
        ax2.tick_params(axis='y', colors='blue', which='both')
        ax2.yaxis.label.set_color('b')
        lgd = plt.legend((acc, valacc, loss, valloss), ('accuracy', 'validation accuracy', 'loss', 'validation loss'),
                         loc='upper center', bbox_to_anchor=(0.5, -0.10), shadow=True, ncol=4)


        plt.title('Model Training Performance', fontsize = 20)
        plt.tight_layout()
        plt.savefig('/home/lborgna/NN/V2Weak/training_plots/train_'+timestamp+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        np.save('/home/lborgna/NN/V2Weak/training_plots/trainingarrays/acc_'+timestamp+'.npy',history.history['acc'])
        np.save('/home/lborgna/NN/V2Weak/training_plots/trainingarrays/valacc_'+timestamp+'.npy',history.history['val_acc'])
        np.save('/home/lborgna/NN/V2Weak/training_plots/trainingarrays/loss_'+timestamp+'.npy',history.history['loss'])
        np.save('/home/lborgna/NN/V2Weak/training_plots/trainingarrays/valloss_'+timestamp+'.npy',history.history['val_loss'])

    if save_roc or precision_recall:
        plt.clf()
        bkgTest, sigTest = datahandler.Loader(bkg_test, sig_test)

        X_Test, y_Test, input_shape = datahandler.AllIn(bkgTest, sigTest, K.image_dim_ordering(), img_rows, img_cols)
        Y_Test = np_utils.to_categorical(y_Test, nb_classes)

        score = model.evaluate(X_Test, Y_Test, verbose = 0)
        Predictions_Test = model.predict(X_Test, verbose = 1)
        Y_Test = Y_Test[:, 1]
        Predictions_Test = Predictions_Test[:, 1]
        rocoutputfile = "/home/lborgna/NN/V2Weak/roc_curves/roc_" + timestamp + ".png"
        if save_roc:
            analysis.generate_results(Y_Test, Predictions_Test, rocoutputfile)
            analysis.save_results(Y_Test, Predictions_Test, timestamp)
            print('Test score: ', score [0])
            print('Test Accuracy: ', score[1])
            print('ROC Curve Saved')
        if precision_recall:
            average_precision = average_precision_score(y_Test, Predictions_Test)
            print('Average precision-recall score: {0:0.2f}'.format(average_precision))

            plt.clf()
            precision, recall, _ = precision_recall_curve(y_Test, Predictions_Test)
            plt.plot(recall, precision, color='b', alpha=0.2)
            plt.fill_between(recall, precision, alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve AUC: {0:0.4f}'.format(average_precision))
            plt.savefig('/home/lborgna/NN/V2Weak/precision_recall/PR_' + timestamp + '.png')
            np.save('/home/lborgna/NN/V2Weak/precision_recall/PR_arrays/precision_' + timestamp + '.npy', precision)
            np.save('/home/lborgna/NN/V2Weak/precision_recall/PR_arrays/recall_' + timestamp + '.npy', recall)
            np.save('/home/lborgna/NN/V2Weak/precision_recall/PR_arrays/aucpr_' + timestamp + '.npy', average_precision)

    if save_cm:
        plt.clf()
        class_names = ['Signal', 'Background']
        normalize = True
        cnf_matrix = confusion_matrix(Y_Test, Predictions_Test.round())
        np.set_printoptions(precision = 2)
        analysis.plot_confusion_matrix(timestamp, cnf_matrix, classes = class_names, normalize = normalize)
        print('Confusion Matrix Saved')


    print("All Done - Timestamp: ", timestamp)
