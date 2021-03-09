# LandCover_Preligens

This is our git for resolving the ENS challenge data : https://challengedata.ens.fr/challenges/48



First we set up a virtual env using conda and we activate it :
 ### conda env create -f environment.yml
 ### conda activate challenge-ens"
 
 (challenge-ens) should appear at the left of your terminal if everything is going correctly.
 
 Then we train the model :
 
 ### python3 framework/train.py --config train_config.yaml
 
 This will create a folder at this path : code/experiments/<day-month-year_hour:minute:second>
 With a save of the model at every epoch in checkpoints, the ID of the validation images, the logs (loss and metric) during the training, some images with their true and predicted masks can be saved in plots if you uncomment the right thing in the callbacks and the according tensors will be saved in tensorboard if you uncomment the callbacks.
 
 This operation will take some time.
 You can change the train dataset in the config file to sample_dataset in order to train on a subsample of the dataset
 
 When the training is over we can predict the class distribution for the images with :
 
  You can choose the model you want, or the set you want to predict in the infer_config.yaml file by modifying the according parameter.
  The xp_file is last per default but you can choose an older training to make your prediction by modifying the line in the config file.
### python3 framework/infer.py --config infer_config.yaml
 
 This will create a file in code/experiments/<date>/ named epoch{nb}_{set}_predicted.csv containing for every image of the set their ID and the proportion of the ten classes.

    
In order to evaluate our model we should run the following command line : (This will work only for val and train dataset as we dont have the label for the test)
    
### python3 framework/eval.py --gt-file train_label.csv --pred-file code/experiments/<date>/epoch{nb}_{set}_predicted.csv -o code/experiments/<date>/epoch{nb}_{set}_predicted_score.csv
    
In order to evaluate on the test set we must submit the epoch{nb}_test_predicted.csv file in the according section at the following url https://challengedata.ens.fr/participants/challenges/48/
