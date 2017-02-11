#!/bin/bash
#%+submission
# ARGUMENTS
# $1   which evaluation step (0,1,2...)
#   $2,3..   any additional arguments we want passed to the main evaluation algorithm (testing, profiling, paths etc)
# ARG    CONTENT                                    DEFAULT
# $2 [input file path ]                             /home/c4c-user/data/input/
# $3 [output file path]                             /home/c4c-user/data/output/
# $4 [code root path, where we unzip to]            /home/c4c-user/data/code/

python predictSpatioTemporal.py $1 $2 $3 $4

# we assume here that "predictSpatioTemporal.py" is in the same directory as current script: predict.sh
# loads $2/X$1.hdf
# makes a prediction ysubg the code found in $3
# saves $3/Y$1.hdf

# Keep this, this if for time stamping your submission
makeready.sh $3/Y$1