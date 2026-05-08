#!/bin/bash

# Number of times to run the main script
NUM_RUNS=4

# Initialize variable to store the job ID of the previous job
PREV_JOB_ID="5869677"

# Loop to submit multiple jobs
for ((i=1; i<=$NUM_RUNS; i++)); do
    echo "Submitting Job $i"
    
    # Add job dependency for sequential execution
    DEPENDENCY=""
    if [ -n "$PREV_JOB_ID" ]; then
        DEPENDENCY="-W depend=afterany:$PREV_JOB_ID"
    fi
    
    # Submit the job and store its job ID
    JOB_ID=$(qsub $DEPENDENCY derecho_training_jsw.sh)
    echo "Submitted Job $i with ID $JOB_ID"
    
    # Update PREV_JOB_ID for next iteration
    PREV_JOB_ID=$JOB_ID
done
