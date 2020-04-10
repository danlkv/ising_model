#!/bin/zsh
#COBALT -n 3
#COBALT -t 50
#COBALT -o hw_job.out
#COBALT -e hw_job.err
#COBALT --debuglog  hw_job.cobaltlog

source $HOME/.zshrc
echo "
Job $COBALT_JOBID
env:
  home: $HOME
  workdir: $(pwd)
  path: $PATH
  pythonpath: $PYTHONPATH
"

python -c "import sys; print('Hello from python', sys.version)"
