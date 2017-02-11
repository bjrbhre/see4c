import pickle
import numpy as np 				# We assume that numpy arrays will be used
import os
from data_io import vprint
import time

class Model():
    def __init__(self, hyper_param=(), path="", verbose=False):
        ''' Define whatever data member you need (model paramaters and hyper-parameters).
        hyper_param is a tuple.
        path specifies the directory where models are saved/loaded.'''
        self.version="Persitent"
        self.hyper_param=()
        self.model_dir = path
        self.verbose = verbose
        vprint(self.verbose, "Version = " + self.version)

    def train(self, Xtrain, Ttrain=[]):
        '''  Adjust parameters with training data.
        Xtrain is a matrix of frames (frames in lines, features/variables in columns)
        Ttrain is the optional time index. The index may not be continuous (e.g. jumps or resets)
        Typically Xtrain has thousands of lines.''' 
        vprint(self.verbose, "Model :: ========= Training model =========")
        start = time.time()
        # Do something
        end = time.time()
        vprint(self.verbose, "[+] Success, model trained in %5.2f sec" % (end - start))

    def adapt(self, Xadapt, Tadapt=[]):
        ''' Adjust parameters and hyper-paramaters with short-term adaptation data.
        Xadapt is a matrix of frames (frames in lines, features/variables in columns)
        Tadapt is the optional time index. 
        Typically the time index has no cuts/jumps and the number of frames is of
        the order of 100.''' 
        vprint(self.verbose, "Model :: ========= Adapting model =========")
        start = time.time()
        # Do something
        end = time.time()
        vprint(self.verbose, "[+] Success, model adapted in %5.2f sec" % (end - start))
         
    def predict(self, Xtest, num_predicted_frames=8):
        ''' Make predictions of the next num_predicted_frames frames.
        For this example we predict persistence of the last frame.'''
        vprint(self.verbose, "Model :: ========= Making predictions =========")
        start = time.time()
        Ytest = np.array([Xtest[-1]] * num_predicted_frames)
        end = time.time()
        vprint(self.verbose, "[+] Success, predictions made in %5.2f sec" % (end - start))
        return Ytest
        
    def save(self, path=""):
        ''' Save model.'''
        if not path:
    		path = self.model_dir
        vprint(self.verbose, "Model :: ========= Saving model to " + path)
        pickle.dump(self, open(os.path.join(path, '_model.pickle'), "w"))

    def load(self, path=""):
        ''' Reload model.'''
        if not path:
    		path = self.model_dir
        vprint(self.verbose, "Model :: ========= Loading model from " + path)
        self = pickle.load(open(os.path.join(path, '_model.pickle'), "w"))
        return self
