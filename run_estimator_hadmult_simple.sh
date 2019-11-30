#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=tf_hadmult

echo "started "`date`" "`date +%s`""

nvidia-smi -L

SING_DIR="/lfstev/e-938/jbonilla/sing_imgs" # Directory with singularity image
SINGLRTY="${SING_DIR}/joaocaldeira-singularity_imgs-master-py3_tf114.simg" # Singularity image

EXE=estimator_hadmult_simple.py # Python script with executing tensorflow
NCLASSES=6 # Number of classes that will be predicted (6 for hadron mult)
BATCH_SIZE=100 # Number of events per batch that will be sent to train and eval
EPOCHS=9 # How many epochs do we want to train?
STEPS_EPOCH=44380 # How many steps cover an epoch?
let TRAIN_STEPS=${EPOCHS}*${STEPS_EPOCH} # Epochs we ant to train in step number
VALID_STEPS=5000 # Number of steps for validation/testing
let SAVE_STEPS=TRAIN_STEPS/10 # How often we want to save checkpoints/models?
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_100mev # Directory where the checkpoint/models will be saved, make sure to use your own dir!
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_75mev
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_50mev
MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/vtx_based_bilinear_50mev
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/ResNet_50mev
#MODEL_DIR=/data/minerva/JLBRtesthad/tensorflow/models/tests

SAVEDMODELS=10 # How many models should we keep (keeps latest)
MODEL=model.ckpt-155330 #"model.ckpt-70584" # model to use for prediction
DATA_DIR=/lfstev/e-938/jbonilla/hdf5 # Directory with our hdf5 files
TRAIN_FILES=${DATA_DIR}/me1Nmc/* # File for training. Can be multiple files. To especify files manually, separete them with a 'space': TRAIN_FILES=FILE1 FILE2 FILE3 etc
EVAL_FILES=${DATA_DIR}/me1Fmc/* # File for validation/test. Can be multiple files. Can especify several files like in TRAIN_FILES
TARGET=hadro_data/n_hadmultmeas_50mev # The name of our training target: 'hadro_data/n_hadmultmeas', 'vtx_data/planecodes', etc
NET=ANN # Neural network architecture. Default 'ANN' is vertex finding
#NET=ResNetX

# We create our MODEL_DIR
if [ ! -d "$MODEL_DIR" ]
then
  mkdir $MODEL_DIR
fi

# String with arguments for training, validation or test
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
#-m cProfile --sort cumulative
nvidia-smi

echo "finished "`date`" "`date +%s`""
exit 0

# Singularity containers
#SINGLRTY='/lfstev/e-938/jbonilla/sing_imgs/LuisBonillaR-singularity-master-py3_tfstable_luis.simg'
#SINGLRTY='/data/aghosh12/local-withdata.simg'
#SINGLRTY='/data/perdue/singularity/gnperdue-singularity_imgs-master-py2_tf18.simg '
#SINGLRTY='/lfstev/e-938/jbonilla/sing_imgs/LuisBonillaR-singularity-master-pyhon3_luisb.simg'
