from dataset import Audio_Dataset

from avalanche.training.supervised import Naive
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics



#TODO Transform the data-set to a avalanche type dataset
#     Test out if nemo works with avalanche, if not create a model with pytorch and implement the transforms
#     Setup functionnality to test multiple CL benchmarks 
#     Enable logging through Sacred
#     

if __name__=='__main__':
    #Import dataset
    DATASET=Audio_Dataset()

    command_train=DATASET(train=True)
    command_test =DATASET(train=False)
    
    # Create Scenario
    scenario = nc_benchmark(command_train, command_test, n_experiences=7, shuffle=True, seed=1234,task_labels=False)
    
    # Create Model
    
    #model = 
    
    # Setup Logging

    ## log to Tensorboard
    tb_logger = TensorboardLogger()

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
    # Training loop

