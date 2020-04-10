#!/bin/bash
#COBALT -n 3
#COBALT -t 50
#COBALT -o hw_job.out
#COBALT -e hw_job.err
#COBALT --debuglog  hw_job.cobaltlog

echo "Hello world, I'm job $COBALT_JOBID"
echo "Working dir $(pwd)"
