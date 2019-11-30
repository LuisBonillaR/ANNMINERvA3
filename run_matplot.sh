#!/bin/bash

SING_DIR="/lfstev/e-938/jbonilla/sing_imgs" # Directory with singularity image
SINGLRTY="${SING_DIR}/joaocaldeira-singularity_imgs-master-py3_tf114.simg" # Singularity image

EXE=matplot.py
PREDICTION=$1
HDF5_FILES=$2
DIS=$3

ARGS="PREDICTIONS HDF5_FILES DIS"

# Show the command to be executed
cat << EOF
singularity exec --nv $SINGLRTY python3 $EXE $ARGS
EOF

# Execute the command
singularity exec --nv $SINGLRTY python3  $EXE
