#!/bin/bash
#COBALT -n 3
#COBALT -t 50
#COBALT -o hw_job.out
#COBALT -e hw_job.err
#COBALT --debuglog  hw_job.cobaltlog

python -c "import sys; print('Hello from python', sys.version)"
