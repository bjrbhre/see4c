#!/usr/bin/env python3

# Usage: python predictSpatioTemporal.py step_num input_dir output_dir code_dir

# SEE4C CHALLENGE
#
# The input directory input_dir contains file X0.hdf, X1,hdf, etc. in HDF5 format.
# The output directory will receive the predicted values: Y0.hdf, Y1,hdf, etc.
# We expect people to predict the next 3 frames.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# The SEE4C CONSORTIUM, ITS ADVISORS, DATA DONORS AND CODE PROVIDERS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL THE SEE4C CONSORTIUM OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon, January 2017

# Use default location for the input and output data:
# If no arguments to this script is provided, this is where the data will be found
# and the results written to. This assumes the rood_dir is your local directory where
# this script is located in the starting kit.
import os
root_dir = os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "results/")
default_code_dir = root_dir
default_cache_dir = os.path.join(root_dir, "cache/")

import time
import numpy as np
from sys import argv, path

def predictSpatioTemporal(step_num, input_dir, output_dir, code_dir, \
                          ext = '.h5', verbose=True, debug_mode=0, \
                          time_budget = 300, max_samples = 0, \
                          AR_order = 1, I_order = 0, MA_order = 0, \
                          num_predicted_frames=8, \
                          save_model = False, cache_data = False, \
                          cache_dir = "", \
                          version = 0.1 ):
    ''' Main spatio-temporal prediction function.
    step_num
        Current file number n being processed Xn.h5.
    input_dir
        Input directory in which the training/adapatation data are found
        in two subdirectories train/ and adapt/
    output_dir
        Output directory in which we expect Yn+1.h5 predictions to be deposited.
        The next num_frame frames must be predicted.
    code_dir
        The directory to which the participant submissions are unzipped.
    ext
        The file extensions of input and output data
    verbose
        if True, debug messages are printed
    debug_mode
        0: run the code normally, using the time budget of the task
        1: run the code normally, but limit the time to max_time
        2: run everything, but do not train, use persistence
        3: just list the directories and program version
    time_budget
        Maximum total running time in seconds.
        The code should keep track of time spent and NOT exceed the time limit.
    max_samples
        Maximum number of training samples loaded.
        Allows you to limit the number of traiining samples read for speed-up.
    Model order
        The order of an ARIMA model.
        Your training algorithm may be slow, so you may want to limit .
        the window of past frames used.
        AR_order = 1 # Persistence is order 1
        I_order = 0
        MA_order = 0
    num_predicted_frames
        Number of frames to be predicted in the future.
    save_model
        Models can eventually be pre-trained and re-loaded.
    cache_data
        Data that were loaded in the past can be cached in some
        binary format for faster reload.
    cache_dir
        A directory where to cache data.
    version
        This code's version.
    '''
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    start_time = time.time()         # <== Mark starting time
    if not(cache_dir): cache_dir = code_dir # For the moment it is the code directory

    path.append (code_dir)
    path.append (os.path.join(code_dir, 'sample_code'))
    import data_io
    from data_io import vprint
    from data_manager import DataManager # load/save data and get info about them
    from model import Model              # example model implementing persistence

    vprint( verbose,  "\n====> STEP: " + str(step_num))
    vprint( verbose,  "Using input_dir: " + input_dir)
    vprint( verbose,  "Using output_dir: " + output_dir)
    vprint( verbose,  "Using code_dir: " + code_dir)
    vprint( verbose,  "Using cache_dir: " + cache_dir)

    # Make a result directory and cache_dir if they do not exist
    data_io.mkdir(output_dir)
    data_io.mkdir(cache_dir)

    # List various directories
    if debug_mode >= 3:
        vprint( verbose,  "This code version is %d" + str(version))
        data_io.show_version()
        data_io.show_dir(os.getcwd()) # Run directory
        data_io.show_io(input_dir, output_dir)
        data_io.show_dir(output_dir)

    # Our libraries
    path.append (code_dir)

    #### START WORKING ####  ####  ####  ####  ####  ####  ####  ####  ####
    vprint( verbose,  "************************************************")
    vprint( verbose,  "******** Processing data chunk number " + str(step_num) + " ********")
    vprint( verbose,  "************************************************")

    # Instantiate data and model objects
    if cache_data:
        cache_file = os.path.join(cache_dir, "Din.pickle")
    else:
        cache_file = ""
    Din = DataManager(datatype="input", verbose=verbose, cache_file=cache_file)
    Dout = DataManager(datatype="output", verbose=verbose)
    M = Model(hyper_param=(AR_order, I_order, MA_order), path=code_dir, verbose=verbose)

    # Read data training frames and train
    if step_num == 0:
        # First time we read the training data.
        train_data_dir = os.path.join(input_dir, "train")
        Din.loadTrainData(train_data_dir, max_samples=max_samples)
        # Train the model
        M.train(Din.X, Din.t) # The X matrix is the time series, the T vector are the (optional) time indices
    else:
        # Reload the already trained model and data (warm start)
        if save_model:
            M.load(path=cache_dir)
        if cache_data:
            Din.reloadData('Din', data_dir=cache_dir, format='pickle')

    # Read additional frames and append them.
    adapt_data_dir = os.path.join(input_dir, "adapt")
    Din.appendSamples(step_num, adapt_data_dir)

    # Save data for future re-use (we do not forget anything at the moment,
    # but this may be waistful in time and memory). We especially may not need
    # the training data.
    if cache_data:
        Din.saveData('Din', data_dir=cache_dir, format='pickle')

    # Adapt the model. We pass all the data we have, the model is supposed to
    # know how to use a window of data in the past.
    M.adapt(Din.X, Din.t)
    # To save the effort of re-computing predictions made by the old model to
    # correct it we could re-load past predictions (still available in the output directory).
    # For simplicity we do not do it here.

    # Eventually save the model for future re-use (warm start)
    if save_model:
        M.save(path=cache_dir)

    # Make predictions
    Dout.X = M.predict(Din.X, num_predicted_frames=num_predicted_frames)
    Dout.t = np.array(range(1, Dout.X.shape[0]+1))

    # Save predictions
    Dout.saveData('Y' + str(step_num), data_dir=output_dir, format="h5")

    time_spent = time.time() - start_time
    time_left_over = time_budget - time_spent
    if time_left_over>0:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Time spent %5.2f sec " % time_spent + "::  Time budget %5.2f sec" % time_budget)
    else:
        execution_success = 0
        vprint( verbose,  "[-] Time exceeded")
        vprint( verbose,  "[-] Time spent %5.2f sec " % time_spent + " > Time budget %5.2f sec" % time_budget)

    return execution_success

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__":
    if len(argv)==1: # Use the default if no arguments are provided
        step_num = 0
        input_dir = default_input_dir
        output_dir = default_output_dir
        code_dir = default_code_dir
        cache_dir=default_cache_dir
        running_locally = True
    else:
        step_num = int(argv[1])
        input_dir = argv[2]
        output_dir = os.path.abspath(argv[3])
        code_dir = argv[4]
        cache_dir=""
        running_locally = False

    execution_success = predictSpatioTemporal(step_num, input_dir, output_dir, code_dir, \
                                              cache_dir=cache_dir)

    if not running_locally:
        if execution_success:
            exit(0)
        else:
            exit(1)


