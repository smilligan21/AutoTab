# AutoTab
Here you will find full interactive code to train Variational Autoencoders for Tabular Data. This code will hopefully be turned into a package soon! For now, it can still be used in the "package" sense. 

Set up your enviorment:

Python / Tensorflow / Numpy
- python version:        3.10.8 | packaged by conda-forge | (main, Nov 22 2022, 08:16:53) [MSC v.1929 64 bit (AMD64)]
- numpy_version:  1.26.4
- tensorflow version: 2.10.0

Reticulate
- #install_miniconda()   #only need to do once
- #reticulate::py_install("tensorflow", force = TRUE)
- reticulate::use_condaenv("r-reticulate")   #run every time
Libraries 
-  library(caret) #for one-hot-coding
-  library(readr) #for importing the data
-  library(dplyr) #for renaming columns
-  library(R6)
-  library(keras)
-  library(tensorflow)
-  library(reticulate)
-  tfa <- import("tensorflow_addons")

Running the code:

Run the full code to ensure all of the functions and in your enviorment. 

Step One: Preprocess your data. 
- If using data of multiple distirbutions I highly encourage you to use min-max scaling instead of standardization. This ensures all the distributions are on the same scale, else the scale of the continous variable   will differ from the binary and categorical.
- Be sure to one hot code your categorical data. 

Step Two: Create your feat_dist
- This will create a key part of the AutoTab code. It will outline the variable, the distribution, and the number of parameters the VAE will produce for the given variable.
- First, run extract_distribution on your orginal data, not the dataset that was one hot coded 
- Then, run feat_reorder
- The goal: feat_dist should be in the same order as your data that is going into the VAE

Step Three: To run the VAE you will need to provide the encoder_info and decoder_info. 
Below outlines the portions of this list and what each indicates:
  - "dense" indicates a dense layer. The next 8 options are as follows:
      second slot: numbr of nuerons, as a numeric number
      third slot: the activation within "". The options are those within the keras package.
      fourth slot: L2 regularization (1), or not (0)
      fifth slot: The regularization rate for L2 regularization
      sixth slot: Batch normalization (1), or not (0)
      seventh slot: Momentum in BN, as numeric number
      eigth slot: Scale in BN,  as numeric number
      ninth: Center in BN
  - "dropout" indicates a drop out later
      second slot: dropout rate, as a numeric number

Step Four: Decide on your prior for the Regularizer. 
- If using  Kullback Leibler divergence this step can be skipped
- If using Mixture of Gaussians (mog) predefine your means, variances and weights.
    EXAMPLE CODE:
          latent_dim = 4
          mog_means <- matrix(c(
            rep(-10, latent_dim),  
            rep(-5, latent_dim),  
            rep( 5, latent_dim),  
            rep( 10, latent_dim)   
                ), nrow = 4, byrow = TRUE)
        mog_log_vars <- matrix(log(0.5), nrow = 4, ncol = latent_dim)  #nrow will always be equal to the number of mog means
        mog_weights  <- rep(1/4, 4) #you need the same number of weigths as there are mog means

Step Five: set your seeds
- You can use the build in function reset_seeds() to set your seeds. Use this for reproducibility and comparing across VAEs when you hyperparameter tune.

Step Six: Run the training step! 
- VAE_train is the function you want to use. Name the final result anything you want. 

training = VAE_train(data =               #Your one hot coded data to which the order and distributional identifiers of feat_dist matches
                  ,encoder_info =         #The build of your encoder with the options outlines above
                  , decoder_info=         #The build of your decoder with the options outlines above
                  ,Lip_en = , pi_enc=     #This allows implementation of Lipschitz continuity in the encoder with indicatoin of how many power iterations, typically only 1!
                  ,lip_dec =, pi_dec=     #This allows implementation of Lipschitz continuity in the decoder with indicatoin of how many power iterations, typically only 1!
                  , latent_dim=           #Your latent dimension
                  , epoch=                #The maximum number of epochs you want the CAE to train. Early stopping will otherwise stop it
                  ,beta                   #The beta on your Regularizer in the ELBO. This is implementation of a BetaVAE
                  ,kl_warm=FALSE,kl_cyclical = FALSE, n_cycles, ratio, beta_epoch=15            #KL Warmup, with options for regular and cyclical along with the number of epochs to do it for 
                  , temperature, temp_warm = FALSE,temp_epoch        #Temperature warming for the gumbel max activation on the categorical data in the decoder 
                  ,batchsize        #Batchsize 
                  , wait, min_delta    #Metric for early stopping, min_delta is the minimum change, wait is how many epochs of this minimal change you will allow
                  , lr                 #Learning Rate
                  ,max_std=10.0        #A scaling factor on the continuous data 
                  ,weighted=0  , recon_weights, seperate = 0        #An option to weight your decoder output, menaing weighting one distribution higher than another, recon_weigths are the weigths, seperate allows                                                                       you to look at the by-distribution loss
                  ,prior="single_gaussian",K =3,learnable_mog=FALSE,mog_means=NULL, mog_log_vars=NULL, mog_weights=NULL         #Implementing the prior 
                  )

                  
To sample from the decoder you must utilize the second set of code named: AutoTab sampling 
- Once you have a sample you will need to reconstruct your data. This will depend on how you preprocessed your data. Be sure to leverage feat_dist to do this! 



