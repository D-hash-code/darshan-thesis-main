'''
Stage One: A Running Model


----- Section A: Initialisation ------

1. Parser

2. Seed and Benchmark. (Master Rank Logging??)

3. Get Device. Set load to memory 'cvt' function

4. Train and Test Loader

5. Build Model
5.1 Set Regularisation Functions
5.2 Set Model(s): NetG and NetD if adversarially training
5.3 Set Optimiser
5.4 Set CNF options and Discriminator options

----- Section B: Training ------

1. Epoch for loop

2. Set Models to .train()

3. Batches for loop

4. Update Learning Rate

5. Cast (real image) Data and Add noise

6. Train Discriminator

Real Data Discriminator Loss
6.1 Zero Grad the network (and optimiser?)
6.2 Create labels tensor for the data
6.3 Run real image data through network for prediction
6.4 Apply criterion(network output,labels) to get loss
6.5 Run loss backward (before this do I need to zero grad netG and optimiserG ? )

Fake Data Discriminator Loss
6.6 Generate noise samples (same number as batch size) and cast to device
6.7 Run noise through generator to get samples
6.8 Create labels tensor for fake images
6.9 Run fake image data through discriminator network for prediction
6.10 Apply criterion(network output, fake labels) to get loss for Disc
6.11 Run loss backward

6.12 Apply Gradient norm to Discriminator parameters
6.13 Apply optimiser step


7. Train Generator Adversarially
7.1 Zero Grad the Network and optimiser
7.2 Run Fake Images through Discriminator, get output
7.3 Calculate loss using, discriminator output, labels and criterion (do I need to apply regulariser loss?)
7.4 Run loss backward
7.5 Clip Grad norm (is this before or after loss backward?)
7.6 Optimiser G step

8. Train Generator Likelihoodly
8.1 Zero Grad Network and Optimiser
8.2 Run compute bits per dimension on real data and model to get bpd, (x,z) and reg states
8.3 Calculate loss = bpd + reg_loss
8.4 Run Loss Backward
8.5 Clip Grad norm and run optimiser.step()

'''

'''
Stage Two: Logging, and Resume

Stage Three: Distributed


'''

# Section A
