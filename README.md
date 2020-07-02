Voted Perceptron
===

An implementation of the voted perceptron algorithm
described in the publication below:

    %0 Journal Article
    %D 1999
    %@ 0885-6125
    %J Machine Learning
    %V 37
    %N 3
    %R 10.1023/A:1007662407062
    %T Large Margin Classification Using the Perceptron Algorithm
    %U http://dx.doi.org/10.1023/A%3A1007662407062
    %I Kluwer Academic Publishers
    %8 1999-12-01
    %A Freund, Yoav
    %A Schapire, RobertE.
    %P 277-296
    %G English

Usage
---
You'll probably have to install some dependencies to run the script.

You can specify the dataset location in `configs.py` as well as 
other self explanatory settings.
In the `data` folder are included [MNIST](http://yann.lecun.com/exdb/mnist/)
and the more challenging [Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
datasets.

Command line parameters:

    -p, --process_count [int]: number of concurrent processes to use to 
        train/test on a dataset
        default to os.cpu_count()

    -mf, --mnist_fraction [0 to 1 with pass 0.0001]: fraction of the dataset
        to train on.
        default to 1
        
    train, test: wether you are training or testing on the dataset

You can train a MulticlassClassifier with the following parameters:

    -e, --epochs [1 to 30 with pass 0.1]: number of times the training set will be repeated.
        If a decimal repeat the remaining fraction of the dataset.
        Note: the epochs preceding the set one will be saved automatically with pass 0.1
        (e.g. if -e 2 epochs 0.1, 0.2, ..., 1, 1.1, ..., 2 will be saved).
        default to 1
    -exp, --expansion_degree: the degree of the kernel expansion
        default to 1

Training files named after the used parameters will be saved in the folder 
specified in `config.py` (default to `./save`).

Example to train on 50% of the dataset for 0.8 epochs with an expansion degree of 2,
with 10 running processes:
~~~
python benchmark/runner.py -p 10 -mf .5 train -e .8 -exp 2
~~~
You can test a saved training file with the following parameters:

    -m, --score_method ['last', 'vote', 'avg', 'rnd']: the method with which the 
        MulticlassClassifier will assign scores to classes.
        default to 'last'
    -f, --filepath: the path to a saved training file

Example to test a training file on 50% of the dataset with score method average,
with 10 running processes:
~~~
python benchmark/runner.py -p 10 -mf .5 test -m avg -f "../save/fashion/filename.pk"
~~~
Information about test and training will be logged in a file
(by default in `logs/events.log`).

Credits
---
Training, class scores, Prediction and multiprocess testing code 
implemented by Marco Di Rienzo for Artificial Intelligence 
class project at University of Florence.

Project structure and class diagram inspired by [bmgee - votedperceptron](https://pypi.org/project/votedperceptron/).