#!/bin/bash
#COBALT -n 128
#COBALT -t 30
#COBALT -o cobaltjobs.64.out
#COBALT -e cobaltjobs.64.err
#COBALT -A QCSim
#COBALT --debuglog  cobaltjobs.debug

module swap PrgEnv-intel PrgEnv-gnu
module load cray-python/3.6.1.1
source $HOME/numpy-env/bin/activate
# export PYTHONPATH=$PYTHONPATH:/opt/python/3.6.1.1/lib/python3.6/site-packages
# cd $HOME/qsim/qsim/run

rpn=1
allranks=64
threads=64

echo "
Job $COBALT_JOBID with size $COBALT_JOBSIZE
env:
  home: $HOME
  workdir: $(pwd)
  path: $PATH
  pythonpath: $PYTHONPATH
"

aprun -n $allranks \
      -N $rpn \
      -d $threads \
      -cc depth \
      -j 1 \
$@

