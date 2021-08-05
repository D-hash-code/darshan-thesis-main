
## Hyperparameters



## Reminder to go through/Later Expansion

- The datasets program in lib
- write_log is set to true only when it is the 'master rank'. Do I need to change this?
- Is there a way I can track memory usage?
- Learning rate and optimisation objective of Discriminator. Should I be using a decreasing learning rate?
- Checkpoint functionality
- Do I want to clip grad norms for adversarial training the generator and discriminator?
- Add transport reguliser loss to adversarial generator
- Do I want to use all evaluation metrics for the test data?
- Resume functionality
- wgan loss
- datadir needs to be decided


## Confusions

- What does a train sampler and train loader do?
- What does the function 'fast_collate' in train.py do?
- Do I need distributed training? Is the distributed training for memory or for multiple gpus?
- Why are they dividing by total number of GPUs for the metrics such as loss??
- How does the number of function evaluations get counted ?
- How do we investigate latent space superiority? Are there other experiments that can provide some inspiration?

## Keys

- Questions: "##**"
- Logging Changes: "##lg"
- Checkpoint Functionality: "##ck"

## Loss
How do I decide on the loss. I have various options on how to apply the loss in the different training types. In particular, for the hybrid network, at each iteration I can either apply the adv loss and lik loss to the generator together or seperately. Furthermore, should I apply the jacobian and transport regularisers to the adversarial losses?

I can apply regularisers to the losses. I can sum and apply losses together or seperately. I can clip gradients or not. Perhaps I can treat each (training type) situation seperately. I can fix the likelihood convention to how they do it in the original. The adversarial training can be informed by GAN convention. I can think about hybrid training later.

## Uncategorised

- Batch Normalisation
- Adaptive learning rate
- Suitable learning rate
- Gradient clipping
- Dropouts

## Data
- Look for papers with latent space experiments

## Model Architecture
- Can I apply style-gan principles?

## Optimisation Objective

## Loss/Cost function

## Evaluation Criteria
