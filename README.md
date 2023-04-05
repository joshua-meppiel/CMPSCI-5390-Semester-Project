# CMPSCI-5390-Semester-Project


## Task 1: Build an overfitting model

The objective in this task is to design a convolutional neural network (not feed-forward neural network) to overfit your dataset. Before working on this task, please watch the lectures in “Chapter 8”, the “Classify MNIST digits using a CNN” and the “Pitfalls in designing CNNs..”, in particular. Also, practice this notebook

https://nbviewer.org/urls/badriadhikari.github.io/DL-2022spring/notebooks/MNIST_using_ImageDataGenerator.ipynb

. to learn how to use an image data generator. It demonstrates how to train a model by loading data from ImageDataGenerators. If you don’t already have one, you will start this phase by creating a similar notebook to load your data. Using all your data (i.e., without splitting), your goal is to design the smallest possible CNN that delivers close to 100% accuracy. Please pay careful attention to the total number of parameters of your model. “Small” refers to the number of parameters, not the number of layers. For example, designing a model with only one Conv2D and one Dense layer will overfit in just a few epochs. But ‘practically’, this is NOT a convolutional neural network because most of the learning/heavy-lifting is done by the Dense layer. For example, you can start with the following model:

```
model = Sequential()
model.add( Conv2D( 64, ( 3, 3 ), activation = 'relu', input_shape = xtrain[0, :, :, :].shape ) )
model.add( MaxPool2D(4, 4) )
model.add( Conv2D( 32, ( 3, 3 ), activation = 'relu' ) )
model.add( MaxPool2D(4, 4) )
model.add( Conv2D( 16, ( 3, 3 ), activation = 'relu' ) )
model.add( Flatten() )
model.add( Dense( 10, activation = 'relu' ) )
model.add( Dense( 10, activation = 'softmax' ) )
```
In your report, you should discuss how the performance (accuracy, precision, recall, etc.) changes when the number of filters and layers are increased/decreased. Also, include your learning curves.

```
# When using generators, you can calculate precision, recall, F1-score, etc. using the following idea (roughly)
Y = [] # empty list of true labels
P = [] # empty list of predictions
for i in range(?):
   x, y = my_generator.next()
   p = model.predict(x)
   Y.extend(y)
   P.extend(p)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(Y, p.round())
```

Debugging tip. You may run into errors while working on this task. Here is a debugging tip: print the type(?), len(?), and/or .shape of all the variables you are working with.

[Requirement for graduate students only] If you provide the output as the input (as an additional channel), what is the smallest architecture (minimum number of layers and filters) you need to overfit the data? Discuss this in your report and submit your notebook. Here is an example code that shows you how to add the output as an additional input channel.

```
# Example of how to use output labels as additional input channel
import numpy as np
N = len(xtrain[:, 0, 0, 0])
L = len(xtrain[0, :, 0, 0])
xtrain_with_outputlabels = np.zeros((N, L, L, 2))
for i in range(len(xtrain)):
   existing = xtrain[i, :, :, :]
   newchannel = np.full((L, L), ytrain_original[i]).reshape(L, L, 1)
   x = np.concatenate((existing, newchannel), axis = -1)
   print(existing.shape, newchannel.shape, x.shape)
   xtrain_with_outputlabels[i] = x
   break
```

If you are using data generators, you can do something like the following to obtain your xtrain and ytrain_original:

```
# Empty placeholders for 1000 RGB images and their labels
mydatax = np.zeros(1000, 256, 256, 3 + 1) # One additional channel for labels
mydatay = np.zeros(1000, 1)
# Read everything from your generator, and fill up the mydatax/mydatay arrays
for i in range(1000):
   x, y = your_generator() # OR your_generator().next()
   # if y is one-hot encoded, you may need to convert y to a single value
   mydatax[i, :, :, :3 ] = x # Existing image in the first three channels
   mydatax[i, :, :, 3 ] = y.reshape(256, 256, 1) # Label value as the last channel
   mydatay[i] = y
```

## Task 2: Split and evaluate on test set

The objective in this task is to obtain highest possible accuracy on the validation set by designing your own CNN. Please note that the reason for randomly splitting ahead of time (and not at runtime) is ‘reproducibility’. Next, train a model using the training set implementing ‘Earlystopping’ and ‘model checkpointing’ observing the accuracy on the validation set. Your goal is to revise your architecture until you find the architecture that yields the highest accuracy on the validation set. Please study how the accuracy changes when the hyperparameters, such as, the number of filters, layers, etc., are increased and decreased. You should also plot your learning curves and include them in your report. Your final evaluation will be on the independent test. Hint: The trick is to create three folders – training, validation, and testing – and then create three separate data generators.

**What to submit?**

1. An HTML version of your ‘annotated’ notebook
    * Your notebooks should be annotated(https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007) with appropriate headings and section names.
    * If you are using Google Colab, please convert the notebook to .html files and submit the .html files, for example, using htmltopdf.
    * Tip for preparing notebooks: read the Ten Simple Rules for Reproducible Research in Jupyter Notebooks.https://arxiv.org/pdf/1810.08055.pdf
2. A PDF report describing your findings (downloaded from your Overleaf project). The reports for the first three phases can be as long as you make them, but the final report has a limit on the number of pages.
3. A link to view your Overleaf project.

**Expectations**

1. You will work on your projects individually (i.e., group submissions are not allowed).
2. Reports for all phases (including the final report) must be prepared using [Overleaf](https://www.overleaf.com/). Non-overleaf submissions will receive a 0 (zero). You are free to use any templates you want. [Here](https://www.overleaf.com/read/vgckqpfdyrwp). is an example. You can learn more about Overleaf [here](https://www.overleaf.com/learn/latex/). If you have accessibility needs, please email me, and I will waive this requirement.


## Task 3: Effects of augmentation

The objective in this task is to increase your accuracy on the validation set (if possible) using data augmentation. With the best model obtained from the previous step, apply various data augmentation techniques (Image generators) and study the improvement in accuracy. A table showing how the various parameters of data augmentation affect the validation set’s accuracy should be reported. Please note that training may take longer to converge after you apply data augmentation.


## Task 4:  Effects of regularization

The objective in this task is to increase your accuracy on the validation set (if possible) using data regularization techniques. Please apply various regularization techniques (Batchnormalization, Dropout, L2 regularization, etc.) and study the improvement in accuracy. A table showing how the various regularization layers/parameters affect the accuracy of the validation set should be reported. Please note that training may take longer to converge after adding regularization layers.

## Task 5:  Use pre-trained models and recent architectures

The objective in this task is to increase your accuracy on the validation set (if possible) using two techniques: a) using more powerful architectures such as ResNet, DenseNet, or NASNet, and b) using pre-trained models such as VGG16 or ResNet50. In your report, discuss how these two methods affect your accuracy. Training (running) times should also be reported.





## Task 6: Final report
In this task, you do not have to perform any new experiments. Instead, you are expected to review and revise your findings of all the previous phases. Following are the guidelines for preparing your final report:

* Your report should not be very long, 10/12 pages at most.
* All tables and figures must be numbered and captioned/labeled.
* Don’t fill an entire page with a picture or have pictures hanging outside of the page borders.
* It is encouraged but not required that you host your project (and report) at Github.
*Turn off the dark mode in Notebook before you copy images/plots (the labels and ticks are hard to see in dark mode).
* Your report should include an abstract and a conclusion (each 250 words minimum).