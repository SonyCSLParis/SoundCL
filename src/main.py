from dataset import Audio_Dataset

from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin,GenerativeReplayPlugin,LwFPlugin,ReplayPlugin,LRSchedulerPlugin,SynapticIntelligencePlugin,GDumbPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.training.templates import SupervisedTemplate
from avalanche.training import JointTraining,Naive

from nemo.core.optim.lr_scheduler import PolynomialHoldDecayAnnealing

from torch.nn import CrossEntropyLoss
from torch.optim import SGD,Adam
import torch

from models import M5
from models import EncDecBaseModel

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorflow.core.util import event_pb2 # Used to interface with
import tensorflow as tf                    # tensorboard logs

import time
import os
import glob
import logging

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


#Setting up the experiment
ex=Experiment('Joint Save')
ex.observers.append(MongoObserver(db_name='Continual-learning'))

@ex.config
def cfg():
    """Config function for efficient config saving in sacred
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_type='adam'
    learning_rate=0.001
    train_batch_size=256
    eval_batch_size=128
    train_epochs=8
    momentum=0.9
    seed=2
    PolynomialHoldDecayAnnealing_schedule=False
    tags = ["Regularization","MatchboxNet","M5","Joint","Naive","Replay","Combined","Architectural"]#to choose from in Omniboard
    save_model=True



@ex.automain
def run(device,opt_type,learning_rate,train_batch_size,eval_batch_size,train_epochs,momentum,PolynomialHoldDecayAnnealing_schedule,save_model,_seed,_run):
    """Main function ran by sacred automain decorator

    Args:
        opt_type (str): Optimizer type
        learning_rate (float): Learning rate
        train_batch_size (int): Train mini batch size
        eval_batch_size (int): Eval mini batch size
        train_epochs (int): Number of training epochs on each experience
        momentum (float): Momentum value in optimizer
        PolynomialHoldDecayAnnealing_schedule (bool): Enable or not the learning rate scheduler
        save_model(bool): Save model as artifact or not
        _seed (int): Random seed generated by the sacred experiment. This seed is common to all the used libraries capable of randomness
        _run : Sacred runtime environment

    Returns:
        int: Top1 average accuracy on eval stream.
    """

    #Import dataset
    DATASET=Audio_Dataset()

    command_train=DATASET(train=True,pre_process=True)
    command_test =DATASET(train=False,pre_process=True)
    
    # Create Scenario
    scenario = nc_benchmark(command_train, command_test, n_experiences=7, shuffle=True, seed=_seed,task_labels=False)
    

    # Create Model
    model= EncDecBaseModel(num_mels=64,num_classes=35,final_filter=128,input_length=1601)#model = M5(n_input=1,n_channel=35)
    
    # Setup Logging

    ## log to Tensorboard
    tb_logger = TensorboardLogger(tb_log_dir='../tb_data')

    ## log to text file
    text_logger = TextLogger(open('../log.txt', 'a'))

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

    # Initialize the optimizer
    if opt_type == 'sgd':
        optimizer=SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif opt_type == 'adam':
        optimizer=Adam(model.parameters(), lr=learning_rate)
    else:
        logging.warning("This type of optimizer is not implemented, defaulting to SGD")
        optimizer=SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
    
    #TODO setup the strategy in a modular way that can be easily modified through the configs of the experiment
    
    #Initialise plugin list.
    #NB: we can add multiple plugins to the same strategy

    plugin_list=[]    
                        # List of used plugins:
                            #LRSchedulerPlugin(PolynomialHoldDecayAnnealing(optimizer=optimizer,power=2.0,max_steps=13260,min_lr=0.001,last_epoch=-1))]
                            #ReplayPlugin(mem_size=50)
                            #SynapticIntelligencePlugin
                            #GenerativeReplayPlugin()


    cl_strategy = JointTraining(
                        model, optimizer,CrossEntropyLoss(), 
                        train_mb_size=train_batch_size, eval_mb_size=eval_batch_size,
                        device=device,
                        train_epochs=train_epochs,
                        evaluator=eval_plugin,
                        plugins=plugin_list
                )

    # Training Loop
    logging.info('Starting experiment...')
    results = []

    # Check if the user requested Joint training
    if isinstance(cl_strategy,JointTraining):
        logging.info("Start of joint training: ")
        res = cl_strategy.train(scenario.train_stream)
        logging.info('Training completed')
    else:
        for experience in scenario.train_stream:
            logging.info("Start of experience: "+ str(experience.current_experience))
            logging.info("Current Classes: "+ str(experience.classes_in_this_experience))

            # train returns a dictionary which contains all the metric values
            res = cl_strategy.train(experience)
            logging.info('Training completed')


    logging.info('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))


    # Logging metrics and artifacts into sacred

    cf_matrix=results[0]['ConfusionMatrix_Stream/eval_phase/test_stream']
    df_cm = pd.DataFrame(cf_matrix/ np.sum(cf_matrix.numpy(), axis=1)[:, None], index = [i for i in DATASET.labels_names],
                    columns = [i for i in DATASET.labels_names])
    fig, ax = plt.subplots(figsize = (24,14))
    sn.heatmap(df_cm, annot=True,ax=ax)
    plt.savefig('heatmap.png')#figure size doesnt work
    _run.add_artifact('./heatmap.png')
    os.remove('./heatmap.png')

    ## Save the model as an artifact
    if(save_model):
        torch.save(model.state_dict(),os.path.join('../models/','model.pt'))
        _run.add_artifact('../models/model.pt')
        os.remove('../models/model.pt')

    ## The save process in tensorboard in multithreaded, we add this sleep to make sure that the file was saved before accessing it
    time.sleep(120) 

    ## Getting the latest file added to ./tb_data
    list_of_files = glob.glob('../tb_data/*')
    latest_file = max(list_of_files, key=os.path.getctime)

    ## Saving the tensorboard data in sacred
    _run.add_artifact(latest_file)# Add raw tf record file to sacred
    _run.add_artifact('../log.txt')
    serialized_examples = tf.data.TFRecordDataset(latest_file)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            _run.log_scalar(value.tag, value.simple_value)

    # We give the average accuracy as the result. The other metrics can be found in Omniboard
    return results[0]['Top1_Acc_Stream/eval_phase/test_stream/Task000']