from dataset import Audio_Dataset

from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin,GenerativeReplayPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.training.templates import SupervisedTemplate

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter

from tensorflow.core.util import event_pb2 # Used to interface with
import tensorflow as tf                    # tensorboard logs

import time
import os
import glob


#TODO Transform the data-set to a avalanche type dataset
#     Test out if nemo works with avalanche, if not create a model with pytorch and implement the transforms
#     Setup functionnality to test multiple CL benchmarks 
#     Enable logging through Sacred
#     


ex=Experiment('Naive Test')
ex.observers.append(MongoObserver(db_name='Continual learning'))

@ex.config
def cfg():
    opt_type='sgd'
    learning_rate=0.001
    train_batch_size=500
    eval_batch_size=10
    train_epochs=1
    momentum=0.9


@ex.automain
def run(opt_type,learning_rate,train_batch_size,eval_batch_size,train_epochs,momentum,_seed,_run):

    #Import dataset
    DATASET=Audio_Dataset()

    command_train=DATASET(train=True)
    command_test =DATASET(train=False)
    
    # Create Scenario
    scenario = nc_benchmark(command_train, command_test, n_experiences=7, shuffle=True, seed=_seed,task_labels=False)
    
    # Create Model
    
    #model = 
    
    # Setup Logging

    ## log to Tensorboard
    tb_logger = TensorboardLogger(tb_log_dir='../tb_data')

    ## log to text file
    text_logger = TextLogger(open('log.txt', 'a'))

    ## print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    # Create the Cl startegy

    cl_strategy = SupervisedTemplate(
        model, SGD(model.parameters(), lr=learning_rate, momentum=momentum) if opt_type=='sgd' else 0,# don't forget to fix
        CrossEntropyLoss(), train_mb_size=train_batch_size,device="cuda" if torch.cuda.is_available() else "cpu", train_epochs=train_epochs, eval_mb_size=eval_batch_size,
        evaluator=eval_plugin,plugins=[GenerativeReplayPlugin()])#we can add multiple CL strategies with plugins

    # Training Loop
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))

    # Logging metrics into sacred

    ## The save process in tensorboard in multithreaded, we add this sleep to make sure that the file was saved before accessing it
    time.sleep(5) 

    ## Getting the latest file added to ./tb_data
    list_of_files = glob.glob('../tb_data/*')
    latest_file = max(list_of_files, key=os.path.getctime)

    ## Saving in sacred
    _run.add_artifact(latest_file)# Add raw tf record file to sacred
    serialized_examples = tf.data.TFRecordDataset(latest_file)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            _run.log_scalar(value.tag, value.simple_value)