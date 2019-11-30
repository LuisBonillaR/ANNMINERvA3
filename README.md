# ANNMINERvA3

This is a Python3 TF framework.

* `estimator_hadmult_simple.py` - Run classification using the `Estimator` API.
* `mnvtf/`
  * `data_readers.py` - collection of functions for ingesting data using the
  `tf.data.Dataset` API.
  * `estimator_fns.py` - collection of functions supporting the `Estimator`s.
  * `evtid_utils.py` - utility functions for event ids (built from runs,
    subruns, gates, and physics event numbers).
  * `hdf5_readers.py` - collection of classes for reading HDF5 (used by
    `data_readers.py`).
  * `model_classes.py` - collection of (Keras) models used here (Eager code
    relies on Keras API).
  * `recorder_text.py` - classes for text-based predictions persistency.
* `run_estimator_hadmult_simple.sh` - Runner script for
  * `estimator_hadmult_simple.py` meant for short, interactive tests.
* `tf_sbatch_minerva.sh` - Runner script for `run_estimator_hadmult_simple.sh`.
  it sends the job to GPUs.
* `matplot.py` Script for making confusion matrices and several more plots after
  prediction.

# Example script to run

```bash
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=tf_hadmult

echo "started "`date`" "`date +%s`""

nvidia-smi -L

# GENERAL
# Directory with singularity image
SING_DIR="/lfstev/e-938/jbonilla/sing_imgs"
# Singularity image
SINGLRTY="${SING_DIR}/joaocaldeira-singularity_imgs-master-py3_tf114.simg"

# Python script executing tensorflow
EXE=estimator_hadmult_simple.py
# Number of classes that will be predicted (6 for hadron mult)
NCLASSES=6
# Number of events per batch that will be sent to train and eval
BATCH_SIZE=100
# Directory with the hdf5 files
DATA_DIR=/lfstev/e-938/jbonilla/hdf5
# The name of our training/validation/testing target: 'hadro_data/n_hadmultmeas', 'vtx_data/planecodes', etc
TARGET=hadro_data/n_hadmultmeas_50mev
# Neural network architecture. Default 'ANN' is vertex finding
NET=ANN
#NET=ResNetX

# TRAINING
# How many epochs do we want to train?
EPOCHS=9
# How many steps cover an epoch?
STEPS_EPOCH=44380
# Epochs we want to train in step number
let TRAIN_STEPS=${EPOCHS}*${STEPS_EPOCH}
# How often we want to save checkpoints/models?
let SAVE_STEPS=TRAIN_STEPS/10
# Directory where the checkpoint/models will be saved, make sure to use your own dir!
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_100mev
# How many models should we keep (keeps latest)
SAVEDMODELS=10
# File for training. Can be multiple files. To especify files manually, separete them with a 'space': TRAIN_FILES=FILE1 FILE2 FILE3 etc
TRAIN_FILES=${DATA_DIR}/me1Nmc/*

# VALIDATION/TESTING
# Number of steps in validation/testing
VALID_STEPS=5000

# model to use for prediction
MODEL=model.ckpt-155330
# File for validation/test. Can be multiple files. Can especify several files like in TRAIN_FILES
EVAL_FILES=${DATA_DIR}/me1Fmc/*

# We create our MODEL_DIR if it does'n exist
if [ ! -d "$MODEL_DIR" ]
then
  mkdir $MODEL_DIR
fi

# String with arguments for training, validation or testing
ARGS="--batch-size ${BATCH_SIZE}"
ARGS+=" --nclasses ${NCLASSES}"
ARGS+=" --train-steps ${TRAIN_STEPS}"
ARGS+=" --valid-steps ${VALID_STEPS}"
ARGS+=" --save-steps ${SAVE_STEPS}"
ARGS+=" --train-files ${TRAIN_FILES}"
ARGS+=" --eval-files ${EVAL_FILES}"
ARGS+=" --target-field ${TARGET}"
ARGS+=" --cnn ${NET}"
ARGS+=" --model-dir ${MODEL_DIR}"
ARGS+=" --saved-models ${SAVEDMODELS}"
ARGS+=" --model ${MODEL}"
# Choose if you are training or testing/making predictions
#ARGS+=" --do-train"
ARGS+=" --do-test"

# Show the command to be executed
cat << EOF
singularity exec --nv $SINGLRTY python3 $EXE $ARGS
EOF

# Execute the command
singularity exec --nv $SINGLRTY python3  $EXE $ARGS

nvidia-smi

echo "finished "`date`" "`date +%s`""
exit 0
```
