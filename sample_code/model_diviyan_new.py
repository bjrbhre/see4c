import pickle
import numpy as np 				# We assume that numpy arrays will be used
import os
from data_io import vprint
import time

class Model():
    def __init__(self, hyper_param=(2,5,), path="", verbose=False):
        ''' Define whatever data member you need (model paramaters and hyper-parameters).
        hyper_param is a tuple.
        path specifies the directory where models are saved/loaded.'''
        self.hyper_param= hyper_param
        # Hyper-parameters :
        # 1 : Number of images/frames to use to compute estimated kernel matrixes' values
        # 2 : Number of pixels to use (size of the weight matrixes) #Must be an odd number>=3, for the matrix to be centered on a pixel.
        try:
            assert self.hyper_param[1]%2==1 and self.hyper_param[1]>2
        except AssertionError:
            self.hyper_param=(2,5)
        self.model_dir = path
        self.verbose = verbose
        self.img_size=32
        self.pixel_weights=[[np.zeros((self.hyper_param[1],self.hyper_param[1])) for j in range(self.img_size) ] for i in range(self.img_size) ]
        self.img_history=[]

    def train(self, Xtrain, Ttrain=[]):
        '''  Adjust parameters with training data.
        Xtrain is a matrix of frames (frames in lines, features/variables in columns)
        Ttrain is the optional time index. The index may not be continuous (e.g. jumps or resets)
        Typically Xtrain has thousands of lines.'''
        vprint(self.verbose, "Model :: ========= Training model =========")
        start = time.time()
        # Do something
        end = time.time()
        vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))

    def get_vect(self,i,j,k,l):
        d=self.hyper_param[1] #Size of kernel matrixes
        T=self.hyper_param[0] #number of frames to take account of

        #Compute coordinates
        x=int(i+k-((d-1)/2))
        y=int(j+l-((d-1)/2))
        try: #Try if out of bounds
            assert x>=0 and x<self.img_size
            assert y>=0 and y<self.img_size
        except AssertionError: # if out of bounds return 0
            return [np.mean(v) for v in self.img_history[:-1]] #Not divide by 0

        # Continue
        return [v for v in self.img_history[:-1,x,y]] # from time=t-1..t-T

    def get_img(self,img,i,j,d): #Get img by filling borders with 0s
        x=int(i-((d-1)/2))
        y=int(j-((d-1)/2))

        try: #Try if out of bounds
            assert x>=0 and x+d<self.img_size
            assert y>=0 and y+d<self.img_size
        except AssertionError: # if out of bounds
            result=np.zeros((d,d)) #Fill blank with 0
            for k in range(x,x+d):
                for l in range(y,y+d):
                    if k in range(self.img_size) and l in range(self.img_size):
                        result[k-x,l-y]=self.img_history[img,k,l]
                    else:
                        result[k-x,l-y]=np.mean(self.img_history[img])
            return result
        return self.img_history[img,x:x+d,y:y+d]

    def adapt(self, Xadapt, Tadapt=[]):
        ''' Adjust parameters and hyper-paramaters with short-term adaptation data.
        Xadapt is a matrix of frames (frames in lines, features/variables in columns)
        Tadapt is the optional time index.
        Typically the time index has no cuts/jumps and the number of frames is of
        the order of 100.'''
        vprint(self.verbose, "Model :: ========= Adapting model =========")
        start = time.time()

        end = time.time()
        vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))

    def predict(self, Xtest, num_predicted_frames=8):
        ''' Make predictions of the next num_predicted_frames frames.
        For this example we predict persistence of the last frame.'''
        vprint(self.verbose, "Model :: ========= Making predictions =========")
        start = time.time()
        T=self.hyper_param[0] #number of frames to take account of
        d=self.hyper_param[1] #Size of kernel matrixes

        self.img_history=Xtest[-T:]  #Load last images

        #Compute kernel weight matrixes
        for i in range(self.img_size):
            for j in range(self.img_size):
                for k in range(d):
                    for l in range(d):
                        self.pixel_weights[i][j][k,l]=np.dot([v for v in self.img_history[1:,i,j]],
                                                                         self.get_vect(i,j,k,l))

                # Rescale according to value at time t
                rescaling_factor=float(self.img_history[-1,i,j])/ max(np.sum(np.multiply(self.pixel_weights[i][j], self.get_img(-2,i,j,d))),0.00001) #Avoid divide by 0
                self.pixel_weights[i][j]=self.pixel_weights[i][j]*(rescaling_factor)


        #Predict images using kernels
        Ypred=np.zeros((self.img_size,self.img_size))
        for i in range(self.img_size):
            for j in range(self.img_size):
                Ypred[i,j]= np.sum(np.multiply(self.pixel_weights[i][j], self.get_img(-1,i,j,d)))

        Ytest=np.array([Ypred] * num_predicted_frames)


        end = time.time()
        vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))
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
