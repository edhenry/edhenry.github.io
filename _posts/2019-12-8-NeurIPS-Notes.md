---
layout:     post
title:      Notes from my NeurIPS 2019 Attendance
author:     Ed Henry
---


# NeurIPS 2019

***DISCLAIMER : THESE ARE RUNNING NOTES FROM MY CONFERENCE ATTENDANCE, I MAY OR MAY NOT COME BACK TO UPDATE THESE NOTES PLEASE USE AT YOUR OWN RISK***


## Quantum Black Workshop

### Reinforcement learning in the real world

The presenter covers the analysis of different Q-learning approaches in the context of their presentation. 

They also cover the importance of evaluating a learned policy and how that can be done.

* We need to understand how well out optimcal policy is able to learn an optimal set of decisions
* We need to propose a set of actions that is realistic for the organization to implement
* We need to interpret out how our optimcal policy differs from the data generating behavior policy.

These questions that we need to ask, outlined above, help us to evaluate the effectiveness of our approaches in the real world.

#### Online and offline policy evaluation

Defn : Evaluating a learned policy through the simulation of the environment

Considerations :

##### Online Evaluation

* Are we even able to simluate the dynamics of the environment?

#### Offline Evaluation

* More easily applied to historical data (we don't have access to a simluation environment)
* Important sampling can be used to estimate the expected reward under a new policy
* Requires estimates of the behavior policy which is usually estimated by separate supervised models

This might be the only way forward as we don't have the capability to simulate the dynamics of the envoironment.


#### Importance Sampling for offline policy evaluation

Goal of PS : estimate the value of a new policy 

Take the ratio of the optimal bahavoior policy at a given time point and * reward at the time step / number of data points -- this generates an estimator for the expected value of importance sampling

[Slide](https://photos.app.goo.gl/TrATMjMK2DKfJyJS7)

##### Pros and Cons

* Has high unbounded variance
* weighted importance sampling has lower variance, but is a biased estimator (because of the weighting)
* Variance is much higher when our policy drastically differs from true behavior
* Variance increases with the length of the episode
* Requires accurate estimate of the behavior policy
* Newer evaluation methods (MAGIC) use both important sampling and model-based approaches depending on the sparsity if your episode

In summary, as time progresses the variance of the policy will increase, this seems intuitively correct

Paper : Evaluating Reinforcement Learning Algorithms in an Oberservational Health Setting

#### Practical Considerations

* Certain domains may **Require** offine evaluation due to ethical or data issues
* OPE may seem relevant in scenarios where a simulator isn't available, but **high variance** makes estimating the value of the optimal policy extremely difficult 
* Investimg time to build a model of the environment is important

## Private Federated Learning

Privacy Principles at Apple for ML

* Data Minimization
	* Collect only _what we need_
* On Device Intelligence
	* Process data local to devices - this prevents the uneeded transport that can allow for eyes that shouldn't see it
* Transparency and Control
	* Allow for 'opt-in'
* Security
	* The foundation for all of this

* Threat Model
	* Malicious Adversarys
		* Perform arbitrary inferences on the learned model to extract data points
	* Curious onlooker
		* Passively looking at updates

* Central Differential Privacy with small privacy parameters (Epsilon <= 2)
	* moment accounting

* Local Differential Privacy 
	* local pertubation on the device modifies the vector into a new vector $$ z = M(w) $$

* Control - allows users to opt into this feature within device settings
	* Expose the information that is being sent to apple
		* Actual parameters, and many other data points

Retention
	* Keep as little data as possible for as short a time as possible
	* If user deletes a data source, immediately remove the corresponding training data, as well.

## CrypTen (Privacy Preserving Machine Learning)

The question being asked "Can data be encypted and still be used to train models"

### Research in Cryptography trying to solve this problem (find slides)

* Homomorphic encryption
	* Data encrypted localled on some device and transmitted back to some central repo
	* Perform some function wrt to data and the function itself can or cannot be encrypted as well

* Secure Multiparty Computation
	* Federal Ownership of data
		* Multiple parties involved in the encryption scheme
		* We can then evalute functions 

* trusted execution environments
	* Much like Enclaves from Intel -- thought these were proven to be insecure with the meltdown and spectre attacks
	* attestation - we can attest to the fact that only the desired function was executred

* Differential Privacy
	* Execute some function on some data that has had noise added to it
		* We have to formally prove that the noise we've added still provides some guarantees through the distributional definition?
			* Need to understand this more.

* Federated Learning
	* Additive noise that has interesting properties described in Secure Aggregation (see Secure Aggregation Protocol Paper)

* How and why does this matter to PyTorch
	* End state of this would be to have a flag in an API where (privacy=True)
	* Far from that. 
	* Built a framework called CrypTen [URL](https://github.com/facebookresearch/CrypTen)

* CrypTen Design Principles
	* Eager Execution 
		* Easier troubleshooting and learning curve
	* Realistic
		* some other OSS projects assumed 2 parties, they wanted to head toward N parties

* Encryption by sharing
	* Multiparty computation
		* Multiple parties process "part" of the data
			* images divded between parties would be done pixel by pixel and this might be uninteresting to any single participant
			* No parties can reconstruct the actual data without collaboration from all parties
	* Assume we have an integer representing some information : $$x \in \{0,..,N-1\}$$
	* Choose some random mask in the same interval : r ~ 
	* Encrypt by subtracting the mask that we've samples above
	* (x-r) becomes independent btween participant clients
	* Decryption in this domain is easy, just add all of the shares
		* just need agreement from all of the participant parties
		* We can design this "agreement" in many different ways
* Encryption is homomorphic
	* adding a public value : $$ [x] + y = (x^(0) + y) $$
	* Multiplication needs triples of encrypted random numbers with the property that $$ [a][b]=[c] $$
		* once we have these tiples we can then generated a share for $$a$$, $$b$$, $$c$$
			* these tiples are sent the participant parties who then calculate epsilon and deltas
			* contain no information because they're all encrypted
			* "open epsilon and delta" - leak no information because they're substracttions from a random number
* Operations for Neural Nets
	* Conbvolutions are additions and multiplies
	* divisions are approximated
	* non-linearities that involve exponentation are 
	* relu break down into sharing 'bits'

* High level architecture [slide](https://photos.app.goo.gl/vQiojJcbSw9tU5TL6)
* Feature List [slide](https://photos.app.goo.gl/gzLXMuHC6Dnh5TiUA)
* Has ONNX integrations

## Bayesian Deep Learning [URL](https://neurips.cc/Conferences/2019/Schedule?showEvent=13205)

* Gradient Descent with Bayes
	* Claim is that we can derive this by choosing Gaussian with fixed covariance
	* Global to local approximation
* Newton's Method from Bayes
	* In this case we choose a multivariate Gaussian [Slide]()[Paper]()
	* We can express in terms of gradient of Hessian of the loss
		* We can use this expectation parameter and ask questions about higher order information about the loss surface
		* If we're not sure about a second order method, then principles say that we shouldn't maybe use these methods
	* 2 parts
		* choose the approximation
		* second is choosing which order of parameters we might want to use
* RMS/Adam from Bayes [Slide]() [Paper]()
	* This has taken the community a long time to figure out but we can see that we can draw lines between the bayeslian learning rule for multivariate Gaussian and RMSprop
* Summary
	* If we add momentum to the bayesian objective a lot of these things can be explained using similar prinsiples

* Bayes as optimization [Slide]() [Papers]()
	* What we can do to derive the bayes rule in this special case
		* We define the loss to be the negative log joint (estimate)
		* We can plug this into an objective function over all distributions
		* With no restriction we should arrive at the posterior distribution
		* Entropy is the negative expected value
		* The expectation of the log ratio of q over e^-l becomes 0 (as log(1) = 0)
		
* Bayes with Approximate Posterior
	* Trying to make a point that using this learning rule, how do we optimize it?
		* Optimizing it in the right way allows us to do much more than variational inference, including exact bayesian inference
	* Bayesian Principle, rather than variational principle

* Conjugate Bayesian Inference from Bayesian Principles [Slide]() [Paper]()
	* Computing these messages in forward backward algorithm, we're finding the expectation of the gradient
	* We can write this loss as two parts
		* loss of the joint
		* depends on the data (conjugate really means depends on the data)
	* Choose a q to match the sufficient statistics 
	* We can write this as a combination of a learning term and a quadratic term
	* Compute the expectation fo the loss, we can see that it's linear in the expectation parameter
	* The expectation of the loss is linear in the expectation parameter
		* Need this to compute the squares (see slide)
	* This is a generalization that applies to things like forward backward, SVI, Variational message passing, etc.
		* This is all proved in the paper link above

* Laplace Appriximation [Slide]()
	* Run the newton method and eventually it will converge to the laplacian 

### Designing New deeo-learning algorithms
#### Uncertainty Estimation with Deep Learning

* Uncertainty Estimation for Image Segmentation [Slide]()
	* We can see missing pieces of sidewalk, etc. this shows wher

* Some Bayesian Deep Learning Methods [Slide]()
	* One line of work proved popular and just keep running standard DL, and then use some ideas to dropout some weights, and doing this it somehow corresponds to solving this bayesian problem (see paper)
		* Pros : Scales well to large problems
		* Cons : Not flexible
	* Point is to get the average with the goal of "how do we choose this average"
		* Get this model, and perturb and this allows to add some noise and allow us to explore a bit
	* The principle of SGD corresponds to a Gaussian with parameters that we cannot really control
	* We can use any parameterization we want

* Scaling up VI to ImageNet [Slide]()
	* Taking a sample of the gradients and this helps us to scale it to ImageNet

* Variation Online Gauss-Newton [Slide]()
	* Improve RMSprop with Bayesian Touch
		* Remove the "local" approximation of the Expectation
		* Add some second order approximation
		* No square root of the scale
	* Takes some more computation but it's worth it in that we're estimation varaince of the diagonal gaussian then we might want to make that tradeoff
	* Estimating a diagonal gaussian with some variance around it, and the variance scaled
	* We can borrow a lot of tricks from the DL side of the world through the framing of the problem we covered previously
* BDL Methods do not really know that they are performing badly under dataset shift
	* This is telling us about uncertainty and about performance
		* We are shifting the data slowly and we can see that accuracy goes down
		* If we're estimating uncertainty it should be reflected in our calibration

* Resources for Uncertainty in DL [Slide]()

* Challenges in Uncertainty Estimation
	* We can't just take bayesian principles and apply them to non-convex problems
	* different local minima correcpond to various solutions
		* local approximations only capture "local uncertainty" -- in the same way that DL only captures a local solution to the functional defn
		* These methods miss a whole lot of the data space
		* This is a very hard problem
	* More flexible approximations really tells us that we need to go beyond second order optimization
		* Fundamentally there are tools that are missing for us to do this

#### Data Importance

* Which examples are more important for a classifier given [Slide]()
	* Does our model really "know' this?
	* Does the model understand why it is the way that it is?
* Model View vs Data View [Slide]()
	* Bayes automatically defines data-importance
	* Points closer to the boundary are more "important"
	* The data view tells us what makes the model certain

* DNN to Gaussian Processes [Slide]()
	* Trained with just a deterministic method (Adam, etc)
	* Can we warp that line to get a distribution?
		* Get a gaussian approximation of this red line and it turns out that the GP are posteriors of linear models
			* posterior is equal to the posterior approximation
		* Find a basis function where this linear approximation and we can convert it to a Gaussian Process
	* These things seem to a dual of eachother

* "Global" to "Local" [Slide]()
	* This posterior approximations connect "global" parameters (model weights) to "local" parameters (data examples)
	* When we use gaussian approximation, we approximate this loss function
	* Local parameters can be seen as "dual" variables that define the "importance" of the data.

* Continual Learning [Slide]()
	* We're not seeing part of the data and when we do this with NN's we show that if we do this in the naive way then we start forgetting the past
	* There is no mechanism to remember the past, this global thing that I want to remember what I did classify and what mistakes I'd made in the past

* Continrual Learning with Bayes [Slide]()
	* Remembers almost everything that happened
	* Computing this posterior is challenging, so we can use posterior approximations

* Some regularization-based continual learning methods [Slide]()

* Functional Regularization of Memorable Past (FROMP)
	* Identify, memorize, and regularize the past using Laplace approximation (similar to EWC)
	*
* Challenges in Continual Learning [Slide]()

* Towards Life Long Learning [Slide]()


## Interpretable Comparison of Distributions and Models

* Divergence Measures
	* Are P and Q the same?
		* We can measure the difference or the ratio of probabilities P - Q or P/Q
		* Divergence measure measuing difference will 
	* Integral Probability Mertrics
		* IPM are looking for a function that is well behaved , meaning smooth
			* differnce in distributional expectations
	* The MMD: integral probability trick [Slide]()
		* Maximimze the mean discrepancy of the distributions
			* smooth function for P vs Q
		* Assuming we can axpress our algorothm using dot products, we can take advantage of the analytical form for solving
		* Infinitely many features that allow us to tell what is different when we use MMD
			* Feature dictionary allows me to distinguish P and Q, no matter the difference between them
		* Expectations of functions are linear combinations of eprected features
			* Turns out the expectation of F is a fot product of F and X
				* 
* How does the Wassterstein 1 behave?
	* Using a wasserstein-1 function that is lipschitz defined

* Phi divergences [Slides]()
	* Taking the ratio of the expectations of the densities 
	* Taking the reverse KL

### Two Sample Testing

* CIFAR 10
	* Weird conclusions in the reults of this paper
	* In the testing of CIFAR 10 - given these distributions how can we measure if they're the same?
	* Remember that MMD(P,Q) = 

* Estimating the MMD [Slide 1]() [Slide 2]()
	* Differnce between the mean embeddings (difference between these latent representations)
	* Expected features of the same size
	* Differences in the mean of the distributions and across the distributions
	* Take an empirical average and we can measure this
	* With this discrepancy, is this MMD true? Small numner "0.09" is small, but not 0.

* Behavior of the MMD [Slide]()
	* P,Q laplace with difference variances in y
	* samples frawn iid from P and Q
	* If we keep drawing on P and Q, we can see that this looks a lot like a normal distribution
	* Asymptotics of the MMD are a normal distribution
	* Central limit theorem results hold
	* Asymptotically normal with a mean at tthe TRUE MMD, and variance sigma^2 of the MMD
		* variance decays asymptotically 
	* What about when P and Q are the same? [Slides]()
		* turns out it's an infinite sum of chi^2 distributions
			* this distribution depends on choice of kernel and what thr distribution of the data is
			* We do know that it converges to distribution of _something_
	* A summary of asymptotics
		* 1. Distributions are close, they're normal
		* 2. The same and its this weird mixture of chi^2
	* Classical statistical tests
		* Distance is big, then we can say they're not the same
		* If the estimate is less than a threshold then maybe they're the same or we didn't have enough data to capture the variance
		* We can take the MMD estimator and ask whether our estimator is bigger than a threshold CL [Slide]()
			* under the known process, when P=Q, we want to reject the null at the rate most 0.05
			* Probability of doing that, to be less than L
		* We can shuffle all examples together, which is a random mixture of dogs and fish [Slide]()
			* we can estimate the distance between these new tilde's and we can estimate what this actually means when P=Q
			* What is the 1 - quantile, and that should be a good estimator
		* Given a kernel, we can now run a test
			* Choosing a kernel, we can start with exponentiated quadratic
				* Kernel is characteristic no matter what bandwidth we pick
				* As we see infinitely many examples, all of the maxx escapes to the right
					* Problem is, we never have infinite samples
					* In our example, bandwidth choice mastters _A LOT_
				* If we choose too smooth of a kernel then we get a witness function that can barely distinguish between the two distributions
			* Power will be really low because of rejection bandwidth will be really high
				* In high dimensions, it doesn't matter what bandwidth we pick because the bandwidth is based on pixel distance between images which breaks down in the curse of dimensionality
			* Often helpful to use a relevant representation, by creating a new representation
				* Take some hidden layer near the end of a classifier (reneralizes a little bit better)
					* Measure MMD between 2000 hidden dimensional representation from a classifer
					* Turns out KID and FID use MMD and give way better properties
				* Interesting that they use the semgentation mask as the pixel count (linear kernel of counts of pixels)
					* This seems super informative
		* What about tests for other distances
			* Sometimes nice closed forms are useful
		* Choosing the best test
			* Picking a kernel that's good for a particular model
			* Power depends on the distributions P and Q (and n)
			* Can maybe pick a good kernel manually for a given problem
	
	* Optmizing MMD for test power [Slide]()
		* As we see more and more data this will converge to Gaussian
			* for large n, the second term is negligible
		* Our estimator is differentiable in kernel parameters
	
	* Data Splitting
		* Important we don't trick ourselves and keep everything statistically bound
		* We need to be looking at the test error here
		* We split part of the data to learn the kernel, and the other part to test that kernel
			* This second part is exactly the standard testing framework covered above
			* This is a methodology notion
		* Learning a kernel is very helpful [Slide]()
	
	* Alternative Approach
		* We can train a classifier to do something like this above
		* We split the data, train a classifier to distinguish X from Y and evaluate it on the other half of the data
			* Accuracy 50% = can't tell, accuracy = 100%, clearly different
				* 60%, the fact that we can classify at all tells us the distributions are different
	
	* Classifier as two-sample test [Slide]()
		* almost exactly equivalent	
		* 0-1 kernel inflates invariance, decreases the test powr

	* Interpretability [Slides]()
		* Can we distinguish two distributions
		* Break up each image into its component pixels and learn a kernel for each pixel
			* Using an ARD kernel
		* We can look at where the witness function "cares about the most"
			* histogram of  witness function might overlap, as the means are close to eachother
			* the points that have the witness function interprets that it looks the most like a dataset
	
	* Main references and further learning [Slide]()
