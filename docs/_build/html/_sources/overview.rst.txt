********
Overview
********

This page talks about the core concepts used in this repository detailing how they work, so you can get started on running your own continual learning experiences.

Avalanche-Lib
-------------

The main Python library used is `Avalanche <https://arxiv.org/abs/2104.00405>`_, an End-To-End library for Continual learning.

The main advantages of Avalanche are:

- Shared & Coherent Codebase
- Errors Reduction
- Faster Prototyping
- Improved Reproducibility & Portability
- Improved Modularity
- Increased Efficiency & Scalability

Core concepts
^^^^^^^^^^^^^

The Avalanche library uses multiple main concepts to enable continual learning with Pytorch:
        
- **Strategies:** Strategies model the Pytorch training loop. One can thus create strategies for special loop cycles and algorithms.
- **Scenarios:** A particular setting, i.e. specificities about the continual stream of data, a continual learning algorithm will face. For example, we can have class incremental scenarios or task incremental scenarios.
- **Plugins:** A module designed to simply augment a regular continual strategy with custom behavior. Adding evaluators, for example, or enabling replay learning.

For more detailed information about the use of this library, check out their main `website <https://avalanche.continualai.org/>`_ and their `API <https://avalanche-api.continualai.org/en/v0.3.1/>`_

Here is an example of a basic training with experience replay taken from their website::

    from torch.optim import SGD
    from torch.nn import CrossEntropyLoss
    from avalanche.models import SimpleMLP
    from avalanche.training.supervised import Naive
    from avalanche.benchmarks.classic import SplitMNIST

    model = SimpleMLP(num_classes=10)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # scenario
    benchmark = SplitMNIST(n_experiences=5, seed=1)

    # evaluation plugins and loggers

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,stream=True),
        gpu_usage_metrics(gpu_id=0,minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger(),TensorboardLogger(tb_log_dir='../tb_data')]
    )

    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=100, train_epochs=4, eval_mb_size=100,
        plugins=[ReplayPlugin(mem_size=100)],evaluator=eval_plugin
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))

Logging
-------
In this project, we leverage the `Sacred <https://github.com/IDSIA/sacred>`_ Python module to ensure comprehensive logging. All logged data is stored in a user-specified `MongoDB <https://www.mongodb.com/>`_ database. 0

The use of Sacred with MongoDB brings several advantages:

- **Comprehensive Parameter Tracking**: With Sacred, every experiment's parameters are recorded, allowing for a detailed overview of each run's setup.
- **Flexible Experiment Replication**: Easily rerun experiments with various configurations, enabling efficient exploration of different settings.
- **Stored Run Configurations**: The database efficiently stores configurations for individual runs, ensuring easy access and comparison between experiments.
- **Result Reproducibility**: By maintaining a log of all the experiments and their respective parameters, you can reproduce any result necessary.

To access and review your experiments in the MongoDB database, we recommend using `Omniboard <https://github.com/vivekratnavel/omniboard>`_. 
To set up the database and Omniboard, follow these steps in your terminal::

    sudo systemctl start mongod.service
    ./node_modules/.bin/omniboard -m localhost:27017:Continual-learning

.. note::
    Make sure to modify the host and database name accordingly in line two to match your specific setup.

How to run
----------

To run the main experiment:

- Run the `setup.sh` script to create the necessary folders for the environment.
- Move into the code directory using `cd src`
- Set the name of your experience and your desired parameters in the `cfg()` function in `main.py`
- Run the experiment using `python3 main.py`