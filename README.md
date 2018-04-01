# mixture_density_network
Mixture Density Network implementation in TensorFlow

### This implementation is designed after the project by Alex Graves described in [this paper](http://arxiv.org/abs/1308.0850).

### What you can do with this code

1. Clean data
2. Train a model
3. ...
4. Profit!

(Currently a WiP)

### Cleaning data

1. Grab the IAM online handwriting data from [here](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database). Full warning, you'll have to create an account to get the data.

Also note that you want to grab **data/original-xml-part.tar.gz** from the top of the list of available downloads.

2. Once you download and extract this data, make a new folder inside of this project's local directory called `data_raw`. Copy the data you've just downloaded into that directory.

3. Clean the data using the exist `data_cleaner.py` script. You can get help on how to use this script by running `python data_cleaner.py --help`. Note that this step can take a while to finish, there's a lot of XML to parse.

This should create a new folder in the local directory called `data_clean`, and inside there should be another folder with a timestamp attached, under which there should be several CSV files with 3-dimensional vectors on each line.

4. Now that you've downloaded and cleaned the data, you can start training a model by running `python train_md.py` with extra args. As before, run `python train_md.py --help` to get help on how to use this script. The only thing that it truly needs to know is the location of the cleaned data (the parent directory of the CSV files that were created in `Step 3`).

Note that the training script isn't configured to sniff out your particular hardware configuration, so it'll run on the hardware corresponding to your TensorFlow installation.

5. The training script will load information about the current run into your Documents folder, under a folder named `mdn_atrocioustimestamp`. In there the training script will log:

* Errors (_error.dat)
* Images of training data input and the network's predictions (deltasplotN.png)
* Images of mixture weights (mixtureweightsN.png)
* Files containing the raw mixture weights for a given input/output sequence (mixtureweightsN.dat)
* A training checkpoint every 500 iterations

### Notes

This script currently has no early stopping criteria, so give it a ctrl+c once you get tired of doing training. Also, there's no concept of epochs in the data procurement; all batches are sampled with replacement from the full dataset.

The memory requirements for this script are relatively hefty; I've had runs take up close to 4 GB of RAM.

There is currently nothing in place for loading a trained model and running it, but the mdn class has methods that allow for recursive sampling from its own distributions (to generate handwriting recurrently).

The future plan is to implement the character windowing feature (perhaps not using the GMM-based windowing technique?) so that handwriting can be generated on a per-character basis.