# RUN

Run stage is to produce data from jobs. 
Since cobalt remembers the script file provided at job submission it is usually best practice to
dedicate directories to jobs with fixed params, like node count.

## Recommended structure

Structure for `i` = number of nodes.

    run/
        i/entry.sh
        i/out.log
        i/debug.log

## Usage of entry script
Example:

    qsub -q $queue_name python ../syntesis/run.py
