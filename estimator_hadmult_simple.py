from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
from absl import logging
import time
import datetime
import tensorflow as tf

from mnvtf.data_readers import make_iterators
from mnvtf.estimator_fns import EstimatorFns
from mnvtf.estimator_fns import serving_input_receiver_fn
from mnvtf.recorder_text import MnvCategoricalTextRecorder as Recorder

parser = argparse.ArgumentParser()

parser.add_argument('--nclasses', default=100, type=int,
                    help='number of classes')

parser.add_argument('--batch-size', default=100, type=int,
                    help='batch size')

parser.add_argument('--train-steps', default=None, type=int,
                    help='number of training steps')

parser.add_argument('--valid-steps', default=None, type=int,
                    help='number of validation or test steps')

parser.add_argument('--save-steps', default=None, type=int,
                    help='number of steps before saving')

parser.add_argument('--train-files', default='', type=str, nargs="*",
                    help='full path to train file. Can be a list of files '
                    'separeted by a space, or using a wildcard ("*")')

parser.add_argument('--eval-files', default='', type=str, nargs="*",
                    help='full path to evaluation file. Can be a list of files '
                    'separeted by a space, or using a wildcard ("*")')

parser.add_argument('--target-field', default='', type=str,
                    help='full path to target data inside hdf5 file')

parser.add_argument('--cnn', default='ANN', type=str,
                    help="CNN architecture to use. Can be 'ANN' or 'ResNetX'")

parser.add_argument('--model-dir', default=None, type=str,
                    help='directory where the model will be stored')

parser.add_argument('--saved-models', default='2', type=int,
                    help='number of models to be saved during training. Just '
                    'the n last models will be kept')

parser.add_argument('--model', default=None, type=str,
                    help='model to use for prediction in model directory. '
                    'If not set, the last saved model/checkpoint will be used.')

parser.add_argument('--do-train', dest='do_train', action='store_true',
                    help='starts script in train mode')

parser.add_argument('--do-test', dest='do_test', action='store_true',
                    help='starts script in test mode')

logging.set_verbosity(logging.INFO)
logging.info("Starting...")


def predict(classifier, data_files, hyper_pars, checkpoint=None,
            cnn_model='ANN'):
    # predictions is a generator - evaluation is lazy
    predictions = classifier.predict(
        input_fn=lambda: make_iterators(
            data_files['test'], hyper_pars['batch_size'],
            mode         = 'predict',
            max_evts     = hyper_pars['test_evts'],
            nclasses     = hyper_pars['nclasses'],
            target_field = data_files['target'],
            cnn_model    = cnn_model),
        checkpoint_path=checkpoint,
        yield_single_examples=True)
    
    logging.info("Writing {}.".format(hyper_pars['predictions_file']))
    recorder = Recorder(hyper_pars['predictions_file'])
    for p in predictions:
        recorder.write_data(p)
    recorder.close()


def evaluate(classifier, data_files, hyper_pars, cnn_model='ANN'):
    classifier.evaluate(
        input_fn=lambda: make_iterators(
            data_files['test'], hyper_pars['batch_size'],
            nclasses     = hyper_pars['nclasses'],
            target_field = data_files['target'],
            cnn_model    = cnn_model),
        steps=hyper_pars['valid_steps'])


def train_no_eval(classifier, data_files, hyper_pars, cnn_model='ANN'):
    classifier.train(
        input_fn=lambda: make_iterators(
            data_files['train'], hyper_pars['batch_size'],
            nclasses     = hyper_pars['nclasses'],
            target_field = data_files['target'],
            shuffle      = True,
            cnn_model    = cnn_model),
        steps=hyper_pars['train_steps'])


def train(classifier, data_files, hyper_pars, cnn_model='ANN'):
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: make_iterators(
            data_files['train'], hyper_pars['batch_size'],
            nclasses     = hyper_pars['nclasses'],
            target_field = data_files['target'],
            shuffle      = True,
            cnn_model    = cnn_model),
        max_steps        = hyper_pars['train_steps'])
    
    eval_spec = tf.estimator.EvalSpec(
            input_fn = lambda: make_iterators(
            data_files['test'], hyper_pars['batch_size'],
            nclasses     = hyper_pars['nclasses'],
            target_field = data_files['target'],
            cnn_model    = cnn_model),
        steps=hyper_pars['valid_steps'],
#        throttle_secs=100
    )
    
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


def main(
    nclasses, batch_size, train_steps, valid_steps, save_steps,
    train_files, eval_files, target_field, cnn, model_dir, saved_models,
    model, do_test, do_train):
    import os

    if (not do_test and not do_train) or (do_test and do_train):
        msg = 'You should use either --train or --test, but not both.'
        raise ValueError(msg)
    
    data_files = {}
    data_files['train'] = train_files
    data_files['test'] = eval_files
    data_files['target'] = target_field
    hyper_pars = {}
    hyper_pars['nclasses'] = nclasses
    hyper_pars['batch_size'] = batch_size
    hyper_pars['train_steps'] = train_steps
    hyper_pars['valid_steps'] = valid_steps
    hyper_pars['test_evts'] = valid_steps * batch_size
    pred_file = 'predictions_'
    pred_file += model.split('-')[-1] + '_' + \
        eval_files[0].split('/')[-1].split('_')[-1].split('.')[0].split('-')[0]
                 
    hyper_pars['predictions_file'] = os.path.join(
        model_dir, pred_file)
    
    cnn_model = cnn
    
    est_fns = EstimatorFns(hyper_pars['nclasses'], cnn_model=cnn_model)

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=save_steps,
        save_summary_steps=save_steps,
        keep_checkpoint_max=saved_models,
        model_dir=model_dir,
        log_step_count_steps=save_steps,
        tf_random_seed=None)
    classifier = tf.estimator.Estimator(
        model_fn=est_fns.est_model_fn,
        params={},
        config=run_config)
    
    t0 = time.perf_counter()
    if do_train:
#        evaluate(classifier, data_files, hyper_pars, cnn_model)
        train(classifier, data_files, hyper_pars, cnn_model)
#        evaluate(classifier, data_files, hyper_pars, cnn_model)
        t1 = time.perf_counter()
        logging.info(' total train time: {}'.format(
            str(datetime.timedelta(seconds=t1-t0))))
    if do_test:
        chekpoint = os.path.join(model_dir, model)
        predict(classifier, data_files, hyper_pars, chekpoint, cnn_model)
        t1 = time.perf_counter()
        logging.info(' total run time: {}'.format(
            str(datetime.timedelta(seconds=t1-t0))))


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

#    exporter = tf.estimator.BestExporter(
#        serving_input_receiver_fn=serving_input_receiver_fn,
#        compare_fn=loss_smaller,
#        exports_to_keep=5)

#        exporters = exporter
