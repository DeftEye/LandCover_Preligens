# LandCover_Preligens

This is our git for resolving the ENS challenge data : https://challengedata.ens.fr/challenges/48
## Presentation of the git :

### First look at the files and folders.
-`environment.yml`: Contain all the conda dependencies for this project.  
-`infer_config.yaml` and `train_config.yaml` : Config files for the prediction and the train.  
-`train_images.csv` and `test_images.csv` : Contain the ID of the train and test dataset.  
-`train_labels.csv` : Contain the true class proportion for every images in the train dataset  
-`code/experiments/` : Contain the result of the training, prediction and evaluation in separate folders named after the date of the start of the train.  
-`data_vizualization.ipynb` : Jupyter notebook in order to see and understand the data.  
-`custom_metric.py` : Small function in order to compute the Kullback-Leibler Divergence between two csv files.  
-`sample_dataset/` : Folder with a subset of the whole dataset in order to test our work. The real dataset should have the same organization inside the folder as this one and the folder should be named "dataset"  
-`framework/` :  Contain the code for this project.  

### Deeper look into framework/
In this folder you will find :  
-`utils.py`, `dataset.py` and `tensorflow_utils.py` : Those files respectively contain usefull functions for yaml files, loading the dataset or manipulate tensor.  
-`model.py` : This file is used to create a model based on the UNet architecture.  
-`train.py` : Load the data, create a model and train it with the parameters in train_config.yaml then save it for later.  
-`infer.py` : Load the model and make a prediction using the parameters in infer_config.yaml then save the prediction for later.  
-`eval.py` : Load the prediction and evaluate them using the Kullback-Leibler Divergence and save the score.  

## Follow this part to make the code work :
### Virtual Env
First we set up a virtual env using conda :  
 `conda env create -f environment.yml`
 
Then we activate it :  
 `conda activate challenge-ens`
 
 (challenge-ens) should appear at the left of your terminal if everything is going correctly.
 
### Training : 
 Then we train the model with this command : 
 
 `python3 framework/train.py --config train_config.yaml`
 
 This will create a folder at this path : code/experiments/<day-month-year_hour:minute:second>.  
 With a save of the model at every epoch in checkpoints, the ID of the validation images, the logs (loss and metric) during the training, some images with their true and predicted masks can be saved in plots if you uncomment the right thing in the callbacks and the according tensors will be saved in tensorboard if you uncomment the callbacks.
 
 This operation will take some time, it took us about 10 hours to train the model with a 1080ti.  
 You can change the train dataset in the config file to sample_dataset in order to train on a subsample of the dataset if it is taking too much time.

### Prediction : 
 When the training is over we can predict the class distribution for the images with :  
 
  You can choose the model you want, or the set you want to predict in the infer_config.yaml file by modifying the according parameter.  
  The xp_file is last per default but you can choose an older training to make your prediction by modifying the line in the config file.  
`python3 framework/infer.py --config infer_config.yaml`
 
 This will create a file in code/experiments/{date}/ named epoch{nb}_{set}_predicted.csv containing for every image of the set their ID and the proportion of the ten classes.  

### Evaluation :
In order to evaluate our model we should run the following command line : (This will work only for val and train dataset as we dont have the label for the test)  
    
`python3 framework/eval.py --gt-file train_label.csv --pred-file code/experiments/{date}/epoch{nb}_{set}_predicted.csv -o code/experiments/{date}/epoch{nb}_{set}_predicted_score.csv`
    
In order to evaluate on the test set we must submit the epoch{nb}_test_predicted.csv file in the according section at the following url https://challengedata.ens.fr/participants/challenges/48/
