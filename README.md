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
You can specify the dataset location in `configs.py` as well as 
reducing the dataset classes.

Command line parameters:

    -p, --process_count [int]: number of concurrent processes to use to 
        train/test on a dataset
        default to os.cpu_count()
        
    train, test: wether you are training or testing on the dataset

You can train a MulticlassClassifier with the following parameters:

    -mf, --mnist_fraction [0 to 1 with pass 0.0001]: fraction of the dataset
        to train on. If greater than on the extra fraction will be repeated.
        default to 1
    -e, --epochs [1 to 30 with pass 0.1]: number of times the training set will be repeated.
        If a decimal repeat the remaining fraction.
        default to 1
    -exp, --expansion_degree: the degree of the kernel expansion
        default to 1

A training file named after the used parameters will be saved in the folder 
specified in `config.py` (default to `./save`).

Example to train on 50% of the dataset for 0.1 epochs with an expansion degree of 2,
with 10 running processes:
~~~
python benchmark/runner.py -p 10 train -mf .5 -e .1 -exp 2
~~~
You can test a saved training file with the following parameters:

    -mf, --mnist_fraction [0 to 1 with pass 0.0001]: fraction of the dataset
        to train on.
        default to 1
    -m, --score_method ['last', 'vote', 'avg']: the method with which the 
        MulticlassClassifier will assign scores to classes.
        default to 'last'
    -f, --filepath: the path to a saved training file

Example to test a training file on 50% of the dataset with score method average,
with 10 running processes:
~~~
python benchmark/runner.py -p 10 test -mf .5 -m avg -f "save/fashion/filename.pk"
~~~

Credits
---
Training, class scores, Prediction and multiprocess testing code 
implemented by Marco Di Rienzo for Artificial Intelligence 
class project at University of Florence.

Project structure and class diagram inspired by [bmgee - votedperceptron](https://pypi.org/project/votedperceptron/).