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

## Bayesian Deep Learning 

[URL](https://neurips.cc/Conferences/2019/Schedule?showEvent=13205)
[Recording]()

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

### Designing New deep-learning algorithms
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

[Slides]()

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

## Representation Learning and Fairness

[Slides]()

* This tutorial will outline how representation learning can be used to address fairness problems
* A Framework for Fair Representation Learning
	* Representation as a fairness problem [Slide]()
		1. Creating a data regulator [Slide]()
			* Determines what the fairness criteria are
			* determines data sources
			* audits results

			* When Training
				* Interacting with all of the stakeholders to understand the fairness criteria
					* output is the fairness criteria
				* Determinining fairness critera
					* Algorithmic fairness
					* Dataset fairness
			* Examples for _how_ to do this
				* Partition the dataset into space of disjoint cells such that similar individuals are in the same cell.
				* Individuals in the same cell should be treated similarly
			
			* Lipschitz continuity implies individual fairness
				* Good news : One can achieve fairness through Lipshitz regularization.
				* Bad news : Data is non-Euclidean (eg. images, graphs, etc). 
					* Standard Euclidean distance metrics aren't a good measure for this
				* Challenge : Can we learn representations of the data such that the l_2 norm is a good metric to compare instances?

			* Group Fairness : Similar Classifier Statistics across groups [Slides]()
			* (im-)possibility result for group-fair classification
				* Classifier statistics are not artbitrarily flexible 
				* eg. Binary classification statistics have two degrees of freesom thus can match two independent statistics across groups
				* Beyond binary classification, the degrees of freedom grows quadratically with number of classes
			* Group Fairness : Advantages and Challenges
				* Advantages
					* Fairly easy to compute and roughly scales with the number of samples
					* Often easier to explain to policy-makeers (as in terms of population behavior)
					* More existing work, strategies already exist for representation learning
				Challenges:
					* data regulator must determine which classifier statistics to equalize.
					* fairness of the representation depends on the quality of the fairness metric chosen by the regulator
					* Group fairness can lead to (more) violated individual fairness, e.g intersectionality can lead to fairness gerrymandering

		* Metric Elicitation [Slide]()
			* Determine the ideal evaluation metric by interacting with users, and experts
			* Query an oracle for this fairness metrics


	* Which statistics should be equalized across groups?
	* Commonly used measures are straightforward functions of classifier performance statistics.


		2. Data Producer
			* Computes the fair representation given the data regulator criteria
		3. Data User
			* Computes ML model given sanitized data
		


## Keynote 1 : How To Know

 [Slides]()

* How can two people living in the same world come to two different conclusions?
* 5 Things Everyone in ML Should know about
	1. Humans continuously form beliefs
		* We don't set them and we're done
		* We continuously update our beliefs
		* Every time we encounter an instance of a bowl, we update our beliefs about bowls
	2. Certainty diminishes interest
		* What you think you know is what determines your curiosity
		* People do not have an accurate model of their own uncertainty
		* If you think you know the answer, you won't check, and if we present the right answer to the person they _still_ reject it
			* Might be why there is confirmation bias
	3. Certainty is feedback driven
		* high level beliefs about concepts
			* most useful for the decision making points in our lives
				* we are sometimes certain when we shouldn't be
		* People learned a novel rule based concept
			* boolean logic to determine daxxy-ness
		* In the beginning there is no concept, it could be a property of the system, or not
		* Entropy of an idealized model has little to do with interest and learning
			* instead this certainty comes from feedback
		* Reasoning about the world
			* Flat earthers -- if they watch online videos that confirm this it might increase the chance of early adoption of this idea as truth
	4. Less feedback may encourage overconfidence
	5. Humans form beliefs quickly
		* Early evidence counts more than later evidence
			* Leads to becoming certain and plays down our ability to update our beliefs
* There is no such thing as a neutral tech platform
	* The order in which information is presented makes a huge difference in our understanding of the world
	* This reinforces some of the studies done around the 2016 elections
	* Children are consuming more online data and this is affecting them


## Veridical Data Science

[Slides]()

* Veridical - coinciding with reality
* PCS Framework for Data Science [Paper]()
	* Predictability (P) (From ML)
	* Computability (C) (From ML)
	* Stability (S) (from statistics)
	* Bridges two of Breimann's cultures
		* PCS connects science and engineering
			* Predictability and stability embed two scientific principles
		* Stability unifies and extends myriad works on perturbation analysis 
			* It's a minimum requirement for reproducibility, interpretability, etc.
			* Tests DSLC by shaking every part (I describe this as wiggling all parts of the system to see how it changes the output)
			* There is always something to follow up when building models
				* New users
				* New patients
				* New collaborators
		* Data Perturbations
			* Data modality choices
			* Synthetic data
			* Data under different environments (invariance)
			* Differential privacy (DP)
			* Adversarial attacks to deep learning algorithms
			* Data cleaning also falls into this data perturbation bucket
		* Model/algorithm perturbations
			* Robust statistics
			* Semi-parametric
			* Lasso and Ridge
			* Modes of non-convex empirical minimization
		* Human decision is prevalent in DSCL
			* Which problem to work on
			* Which data sets to use
			* How to clean
			* Whats plots
			* What data perturbations, etc.
			* WRITE THIS ALL DOWN (MODEL CARDS FOR MODEL REPORTING)
		* Reality correspondences <- great description of what we do when we "model" something
		* How do we choose these perturbations?
			* One can never consider _all_ perturbations 
			* A pledge to the stability principle in PCS would lead to null results if too many perturbations were considered
			* PCS requires documentations on the appropriateness of all perturbations
		* Expanding statistical inference under PCS
			* Modern goal of statistics is to provide one source of truth
		* Critical examination of probabilistic statements in statistical inference
			* Viewing data as a realization of a random process is an ASSUMPTION unless randomization is explicit
				* THIS DATA COULD HAVE BEEN GENERATED NON-RANDOMLY
			* When not, using an r.v. actually implicitly assumes "stability"
			* Use "approximate" and "postulated" models
		* Inference beyond probabilistic models
			* We need to have a way to bring PDE models in with things like synthetic data
			* Proposed PCS framework
				* Problem formulation - translate the domain question to be answered by a model/algorithm (or multiple of them and seek stability)
				* Prediction Screening for reality check : filter models/algorithms based on prediction accuracy on held out test data
				* Target value perturbation distribution - Evaluate the target of interest across "appropriate" data and model pertubations
	* Making Random Forests more interpretable using stability

## Uniform Convergence may be unabe to explain generaliation in deep learning 

[Slides](http://www.cs.cmu.edu/~vaishnan/talks/neurips19_uc_slides.pdf) 
[Poster](http://www.cs.cmu.edu/~vaishnan/talks/neurips19_uc_poster.pdf)

* Tightest uniform convergence bound that eventually shows it is vacuous
* Given a training set $S$, algorithm $h$ in $\mathbb{H}$, then [Sl]
* In what setting do we show this bounds failure/
	* Separating an nested hyperspheres, with no hidden noise, and completely separable
	* Observe that as we increase number of training data point, the loss follows as expected
	* As we change the label of the datapoints between the hyperspheres, we take the set of all data points and show that s' is completely mis-classified even though it is a completely valid member of the training dataset.
		* intuitively this can happen only if the boundary we have learned has "Skews" at each training point
			* What this means is the learn decision boundary is quite complex
			* This complexity that even the most refined hypotehsis class is quite complex
			* This proves the bounds are vacuous
		* This overparameterzed deep network can 
	* Looking aheead it's important to understand the complexities contained within the decision boundaries and derive new tools as a test case

## On Exact computation with an infinitely wide neural network [Slides](https://neurips.cc/media/Slides/nips/2019/westexhibitionhallc+b3(10-10-05)-10-10-20-15845-on_exact_comput.pdf) [Poster](http://www.cs.cmu.edu/~ruosongw/poster_cntk.pdf)

* Neural Tangent Kernel's (NTK's)
* Theoretical contribution
	* When width is sufficiently large (polynomial in number of data, depth and inverse of target accuracy) the predictor learned by applying gradient descent 
* Empirical contribution
	* Dynamic programming techniques for calculating NTK's for CNN's + efficient GPU implementations
	* There is still a gap between the performance of CNN's and that of the NTK's
		* This means that the success of deep learning cannot be fully explained by NTK's
	* Future directions
		* Understand neural net architectures and common techniquest from the lends of NTK's
		* Combine NTK with other techniques in kernel methods

## Generalization Bounds of Stochastic Gradient Descent 

[Slides](https://neurips.cc/media/Slides/nips/2019/westexhibitionhallc+b3(10-10-05)-10-10-25-15846-generalization_.pdf) 
[Poster](https://drive.google.com/drive/folders/1dHLGUG78Uei4c8YEVfv2qFSVO8MGfDy5?usp=sharing)

* Learning large overparameterized DNN's, the empirical observation don extremely wide networks shows that generalization error tends to vary
* Deep RELU networks are almost linear in terms of their parameters on small neighborhoods around random initialization
* Applicable to general loss functions
* Generalization bounds for wide and DNN's that do not increase in network width
* Random feature model (NTRF) that naturally connects over-parameterized DNNs with NTK

## Efficient and Accuract estimation of lipschitz constands for DNN's 

[Slides](https://neurips.cc/media/Slides/nips/2019/westexhibitionhallc+b3(10-10-05)-10-10-30-15847-efficient_and_a.pdf) 
[Poster](https://www.seas.upenn.edu/~mahyarfa/files/Slides_NeurIPS_2019.pdf)

* Lipschitz constant means with 2 points X and Y, they'll be close before and after being passed through the neural network
	* generalization bounds and robust classification lean on this
	* This problem of computing Lipschitz constants is NP hard so we try to find tight bounds around this
	* Say we have an accurate upper bound of a model
		* We can take a point f(x), and we can measure \delta of mis-classification and input that back into the network
			* We can certify that if we perturb X in a small ball drawn around this delta, it odesn't change the classification
			* If we can find this small lipschitz constanc we might be able to prove that this network has a form of robustness
		* HOw do we do this?
			* Product of the norm of the matrices
				* Simple methods like this give upper nounds to the lipschitz constant that are conservative. Can we do anything more accurate?
			* We cna frame up finding this Lipschitz constant as a non-convex optimization problem
				* Over approximate the hidden layers via incremental qudratic constaints
					* Give rise to a semi-definite program giving us this tight upper bound that we're looking for
				* We can trade off scalability with accuracy of the upper bound
		* How does this bound we get compare to others?
			* We show that in general our bound is much tighter than other bounds
	* Adversarial robustness
		* Hypothesis trained using adversarial optimizers
			* Emperitically when we evaluate this lipschitz constant these networks have much lower
* Accurate and scalable way of calculating Lipschitz constants in Neural Networks

## Regularization Effect of a large initial learning rate 

[Slides](https://neurips.cc/media/Slides/nips/2019/westexhibitionhallc+b3(10-10-05)-10-10-35-15848-towards_explain.pdf) 
[Poster](https://drive.google.com/file/d/1lg8hg-1QMFUDvYzZWg83DF7X3yXSl9kw/view?usp=sharing)

* Large initial learning rates are crucial for generalization
* Scaling back by a certain factor at certain epochs
	* small learning rates early on lead to better train and test performance?
* Learning rate schedule changes the order of patterns in the data whcih influence the network
	* class signitures in the data that admit what the class is, but it will ignore other patterns in the data
* Large learning rates initially learn easy patterns but hard-to-fit patterns after annealing
* Non-convexity is crucial because different learning rate schedules will find different solutions
* Artificially modify CIFAR 10 to exhibit specific pattern types
	* 20% are hard to generalize - because of variations in the image
	* Easier to fit in the second set 20%, easy to generalize but hard to fit -- this is by construction
	* Path that imitates what the class is
	* 60% of examples overlay a patch on the image and the memorization of the patch early on shows this method fits early and doesn't generalize well

## Data-Dependent Sample Complexities 

[Slides](https://neurips.cc/media/Slides/nips/2019/westexhibitionhallc+b3(10-10-05)-10-10-40-15849-data-dependent_.pdf) 
[Poster](https://drive.google.com/file/d/1E5SV6Mx_YDCPqnE5aAISSTk09mUkNu6L/view?usp=sharing)

* How do we design principle regularizers for DNN's
	* Current technqiues are designed ad-hoc
		* Batch-norm and dropout - we know they work, but noy why
	* Can we prove a theoretically upper bound on the generalization and hope it improves performance
* Bottle-neck prior
	* most priors only ocnsider the norms of weight matrices and because of this they get pessimistic bounds that are exponential in depth
* Bounds that depends more on data dependent properties
	* upperbounded by the weights and training data
	* informal theory is that this can be upper bounded by (see slides)
	* Jacobian norm isthe max norm of the jacobian of the model on the hidden layers
	* margin is the largest logit of the output, minus the second largest
* INterpretation of this bound is it measures the 'Lipschitzness" of the network around training examples
* Noise stability is small in practice with looser bounds (see slides)
* Regularize the bound
	* penalize the square jacobian norm in the loss 
	* Normalzation layers such as batch norm and layer norm
		* Helps in a lot of settings 
* Check the bounds correlate with the test error and we found that our bound correlates well with test error
* COnclusions
	* Tighter data dependent properties
	* Bound will avoid the exponential dependency on th depth of the network and optimizing this bound helps to improve performance
	* Follow up work : tigher bounds and empirical improvement over strong baselines

## Machine Learning Meets Single Cell Biology : Insights and Challenges

[Slides]()

* Address something asked by DaVinci - How are we made?
	* We're created from a single cell and it eentually creates every cell in our body.
	* How does this process happen?

* Cells are like tiny computers, taking input and output through things like proliferation, differentiation, and activation
* ALl cells have the same genetic code -- 3 billion letters
	* How our genome is the instruction set for assembling the different cells
	* Telling eachother how to behave
	* We have 100's of cells
* Single cell RNA-sequencing in droplet micro-fluids
	* Measures for every single gene and cell for what gene it is and what cell it came from
* The data matric for one sample ~ 250 million
	* Gene by cell matrix that is rife with errors and artifacts from measurement
	* We only capture about 5% of the transcripts, or what the humans are expressing
* In the field, zero-inflation has taken root
	* It's wrong
	* "Drop out" -- this is uniform sampling and it sometimes leaves us to capture no gene or transcription gene
		* every sample is affected by this
		* No value is at it's actual value
		* This should be modeled properly and not with 0 inflation
	* How do we handle all of this data
		* We like to visualize into 2 and 3 dimensions
		* PCA failed this data type and we couldn't visualize it very well
	* Following a Keynote at NeurIPS, someone presented T-SNE and it seems to fit the data well and we get good cell level separation
	* While we might have a matrix of 20000 genes x 100000 cells -- T-SNE and UMAP seem to capture this non-linearity well
	* We have this manifold because cell phenotypes are highly regulated
		* We can see in 3D the nicely shaped non-convex shape
		* Similar shapes in families because few regulators drive cell shape
		* Lots of feedback loops and interactions between these genes, which limits and constricts the phenotypes a cell can be in
		* Still have challenges in visualization
			* Build better vis for this data as it's non-uniform and 5 orders of different density 
				* Challenge is to handle this data with such different densities, it trips up many of the approaches we have today
		* Common way to viz beyond 2 or 3 dimensions using nearest neighbor graphs
			* We connect a cell to cells nearest that cell
				* This is dependent on probability distributions which help to define this similarity metric
		* The idea is one we have this graph we can really retain the manifold and use things like geodesic differences 
			* Each tiny differnce in this graph can represent small differences between cells
				* We can do distances and walks in these grpahs that allow us to measure the distance between cells
		* Data is extreme structures and there are communities
			* Social media community detection approaches find cell states and cell densities that are captures as cliques in the graph
		* Nice thing is that these graphs are connected
			* They share connectivity which allows us to cpature cell type transitions 
				* These transitions are very sparse relative to the cell type
		* Using 100 ro 200 examples over 10000 entities
			* This isn't regular science and this works because biology has lots of structure and isn't adversarial in that respect
			* 100000 or 1000000 cellshave awesome things -- treating each cell as a computer we can assume that the mollecrular influences create statistical dependencies in the data
		* Out of the box algorithm gets us a correct reconstruction with no prior information of TCell networks
		* Thisallows us to do disease regulatory networks and helps us to understand what is wrong in this specific cancer patient
	* Asynchronous nature of the data
		* All of our immune cells are in our bone marrow and these cells are able to generate all varities of immune cells within our bodies
		* Asynchrony enables the inference of temporal pgoression
		* From a single time point we can capture all of the dynamics of the process
	
	* Pesudo-time
		* Reconstructing developement which allows us to reconstruct order grom a single time point
		* THis process is highly non-linear we can order cells by chronology and the assumption is cell phenotypes change gradually
	`	* Cannot be treated as absolute values as we know this data is incredinly noisey
	* Wanderlust 
		* We are able to reconstruct accurate progressions and discover order and timing of key events along differentiation
		* This checkpoint of DNS recombinations inside of a cell, we wanted to understand if it were OK
			* pediatric cancer is caused by understanding this checkpoint
			* This wasn't known until we could find this tiny new cell population and it's novel regulation
	* Data is structured 
		* bifurcations through use of walks and waypooints
			* the direct route between two cells along the same path should be more direct than non-immediate connections
		* These waypoints help to resolve structure
			* We find these using spectral clustering
* Mapping development
	* We want to order these cells on their manifold and understand how they bifurcate
	* What decision making is going on and what is their possible future cell types and propensity to turn into these cell types
	* Palantir : Building a Markiv Chain out of this graph allows us to find time ordering in our neighbor grpahs
		* strong assumption that development goes forward and not back
		* broken in processes such as cancer
			* build a directed graph from this
		* we can look at the extrema states and we find ourselves with an absorbing markov chain
		* This allows us to compute the end states of all of our cells and we can roll out the fate for each cell
			* entropy of these probabiities for all cells
		* The proof is in the pudding -- applied to early mouse developemnt (endoderm (all internal organs are made of this))
		* Data organized nicely along these dimensions
			* cells aligned along temporal orders
			* approximal distral organizations
			* These organizations happen head to tail
			* A smooth gut tube, even though we can't see anything that accounts for this organization, we can see the primodal organs jutting out from this tube
		* We can transcriptionally see where cells are headed a full day before they progress in that way
			* We go into the early days of the first decisions of the cells
				* cell can become one of many classes
		* We can take spatio-temporal maps of the mamaallian endoderm
			* We see when FGR1 and FGR2 are both high, they'll be primitive endoderm
		* Very high entropy _Right_ before this deicsion is made and entropy drops immedaitely after
			* analysis shows that biologist saw that these cells are plastiq -- they can change by jumping out of that area and into the emryonic layer and assume the nature of the other cells
		* Plasticity was predicted computationally, and we were then able to verify empirically

* The Human Cell Atlas
	* Cells in our body, relationships between then, and transitions that happen within the human cellular system
	* Most of the data is still single cell genomics
		* This atlas will have single cell genomics and spacial information of these cells
	* This will require tons of computation
		* global and open community that anyone can join
		* public data of 10 billion cell playground
	* Human cell atlas will serve as a healthy reference and ground truth for disease
		* The methods we have now don't scale
		* Data harmonization
			* data from multiple samples that might be diseased
				* our methods mistake disease for biological differnce
		* Factor analysis for good gene programs
			* How this data factors betwen cells and genes
			* Simply comparing disease to normal
* Latent sapces : Deep Learning in scRNA-seq
	* count basis projected into latent space
		* Data denoising and integration
		* low dimensional embeddings
	* Interpretation of latent factors is still lacking

* Our goal is not to predict, but to understand
	* Often the outlier is the _most_ important
	* machine learning is all about the common mechanism and not the outlier, whereas biology _wants_ to know those outliers
	* Keep our eye on the goal in biology and understanding that something rate

	* Dendritic cells are rare 
		* These cells split into different 
	* Cell types aren't necessarily clusters
		* Though clusters still have their own version of structure to them
	* The more we zoom in, the more we find structure in this data and we see that meta-cells have real peaks in their density
		* These meta-cells are defined by different programs and different covariances
* Acute myioloid lukemia is accute cancer gone awry
	* Normal immune cells seem to overlap
	* These are early projectors of cancer cells -- before they go awry and crazy
	* We want to know what happens that normal <> breaks, and cancer forms
	* When we look at classicial methods, these diseases don't connect
		* We believe in covariation and find a manifold that is driven by covariation and not just normal distributions
		* Covariation in much lower dimensional space is much more computationally efficient
		* The regulatory systems that go awry
			* When we run these methods, we can see exactly where the cancer breaks off and becomes cancer
* Response to therapy
	* Bone marrow transplant patients who relapse and understanding how that relapse happens
	* Understand the immune populations that differ between them, and using these dynamics we can see cell populations that really follow and raise up as the tumor burden rises and falls
	* These are very tiny populations, so one has to be very careful in computation

* Epigenetic data	
	* What potential regulators that can be regulating these systems
	* We can build generative processes, and using these latent variables we can understand different properties of these biological systems
		* What is the covariate nature
		* We can understand inter-variable influence
		* What factors combine to what targets through their regulators
* Most cells in a tumor in solid tumors are not cancer
	* immune cells and supportive tissue make up 90%
	* using factor analysis we can see cancer highjacks early development of these processes
		* using a program that the embryo knows to metasticize a new organ in another part of the body
			* understanding how they survive in the brain
			* Identify that all cancers, both breast and lung, have the same gene that created their ability to survive there
		
	* Cancer uses regenerative mechanisms for it's evil deeds
		* 1 change in 6 billion base pairs can make it go different under injury
		* The reason that this is is because there is enormous cross talk and remodeling between the eputhelial systems and the cancer
	* Spacial techniques are critically important
		* Rapid autopsy programs allow us to collect samples

## Test of time award : Dual Averaging Method for Regularized Stochastic and Online Optimization

[Slides]()

* Stochastic optimization and empirical risk minimization
* We want to minimize the empirical risk of a very large sample
	* Convergence theory 
		* Depending on loss function's convexity or strong convexity
	* Online Convex optimization
		* Here we consider an online game where each player predicts t+1
			* suffers a loss $f_t(w_t)$
			* loss measures total loss of a fixed $w$ from hindsight
			* very similar to SGD
	* Compressed sensing / sparse optimization
		* LASSO
			* minimize quadratic function with a constraint in the $l_1$ norm
	* Proximal gradient methods
		* Adding up of convex

## Causal Confusion in Imitation Learning 

[Slides]()

* Imitation learning is a powerful method for learning complex behaviors
	* driving cars, flying drones, and grasp and pitch
	* Behavioral cloning
	* Supervised learning through observations of experts
	* Not perfect
		* Expert state and we roll out these states we get errors of the imitator acculator that show up in other parts of the state space
		* distributional shift that arises due to this causality
	* Does more information lead to better performance?
		* What happens under distribution shift?
		* Turns out that a given model learns to pay attention to the road and brakes when someone is in the road
	* These fail because the model cannot infer causality
	* Can we predict the expert's next action
		* This is the only cause
		* The expert ignore it and its nuisance variable
		* End state variables
			* If we learn 1 imitator that watches the road
			* We can learn another imitator that wrongly treats both variables as a cause
		* IN general if we have 2^N possible causal graphs
	* Existence of causal confusion
		* Consider 2 examples
			* learns actions through history and one that doesn't
			* validation score on held out data history plays a role in working well with history but in test it causes confusion
		* How do we demonstrate this
			* We add to the original state and use this information to create confounded states
			* This corresponds to having just the causal 
			* Use a VAE and treat the dimensions of the latent variables as potential causes
				* Using behavioral cloning on the original state we get expert like rewards
				* On confouded states, we do much worse indicating causal confusion
			* What we need is to have a causal graph that indicates the random variables that the expertt pays attention to
			* In the first phease we learn from all possible causal graphs 
				* Binary vectors 1 - cause, 2 - nuisance
				* randomly sample a causal graph and mask out the nuisance part of the state
				* we concatenate the graph and feed it to into a NN and it predicts an action
					* behavioral cloning loss
			* In the second phase we infer the correct causal graph
				* intervention changes th distribution of the state
				* we score all possible graphs on additional information
					* Mode 1 : query reward
					* Mode 2 : 
			* Collect trjectories as policies
			* Query expert on states
			* Pick graph most in agreement with experts
		* DAGGer baseline performs significantly worse
	* Learned graph visualization
		* learned causal graph can ignore nuisance variables
		* More information can hurt performance without this effort
		* How to scale this up to more complicated tasks

## Imitation Learning from Observations by Minimizing Inverse Dynamics Disagreement 

[Slides]()

* MDP formulation 
* Divergence minimization perspective on inverse learning
	* GAIL or AIRL 
		* KL and JS divergences could be used
		* Adversarial training for divergence minimization

## Structure Prediction approach for generalization in cooperative multi-agent RL 

[Slides]()

## Guided Metapolicy policy Search 

[Slides]() 
[Paper]()

## Using logarithmic Mapping to Enable Lower Discount Factors in Reinforcement Learning

* Standard RL Setting
	* Discount factor in play here
* This can be modeled as an MDP
* Goal : find an optimal policy to maximize reward which is some long term objective

## Weight Agnostic Neural Networks 

[Slides]() [Poster]()

* Innate abilities in animals
	* we're beginning to understand that ML architectures seem to have innateness
	* CNN's are so well suited to image processing that they can do many tasks in that area
* How far can we push this innateness idea
	* To what extent can NN architectures along encode solutions to tasks?
	* Different kind of NAS
		* WE're looking for NAS' that perform without any training at all
		* Judged on 0-shot performance
		* Because of the large weight space
			* Use a single value for our weight space
			* Judge how well the network works by doing several roll out with different values
		* Create a population of minimal networks
			* These have inputs with no hidden nodes, connected to some outputs
		* Performance of the network is averaged over rollouts and then ranked
			* vary the networks to create new populations and continue the process
		* We can vary in one of 3 ways
			* Inset node
			* Add hidden connection
			* Change the activation function
				* Gauss
				* ReLU
				* Sigmoid
			
	* Tested on 3 RL tasks
		* cart-pole
		* bi-ped
		* car racing
	* Compare these WAN found topologies
		* when trained can reach SoTA
	* Randomize the weights they don't perform well
	* Shared weights produce pretty good behaviors
	* If we tune the weights, they perform the same kind of SoTA performance as general purpose networks
		* Able to do this with much smaller networks, sometimes orders of magnitudes
	* For fun we tested this on MNIST
		* Here we use a random weight -- we get an expected accuracy of 82%, best single weight is 92% -- little better than linear regression
	* Searching for building blocks toward a differnt kind of neural architecture search
		* architectures that have innate biases and priors
	

## Social Intelligence

[Slides]()

* What has changed since Rosenblat started playing with NN's?
	* Compute power
	* Data
	* This gave Dean the idea to found Google Brain
		* Early 2010's
	* Edge TPU - 2 watts and 4 TOPS
		* Mobile-net-v2 @ 400FPS
	* Coral.ai prototyping boards
	* Running workloads locally is important, BUT NOT SUFFICIENT, tool for implementing and reasoning about privacy
	* Have to be "smart" about what data is thrown off of a device using ML
	* Sanity wrt energy consumption and other natural resources
		* moving data is what costs energy
		* once the data is in a register and we want to operate over it, it's relatively free
			* moving it is what costs energy
* How do we make these giant distributed frameworks run efficiently, effectively, and privately?
	* Federated learning allows centralized inference but localized training
* Scale?
	* 100's of millions of android phones have machine learning running on them
	* Scale story is more complex
		* data are both abundant and rare / precious
			* we don't always have access to the data
		* compute is both massive and limited/precious
			* don't effect UX
		* Premium on quick covergence, ie/ < 1 pass over the data
		* FL both enabled ML fairness and can be in tension with it
			* long tail and rare event learning can endanger some of the privacy aspects
	* Generative models are really important in the federalted setting -- and it's not just about making pretty pictures...
		* Capturing this underlying data generating distribution is key
* Open problems
	* tightening bounds, extending domains, handling infrastructure
* Where is all of this AI stuff going anyway?
	* ML / Data Science
		* Conflicting narratives
		* generally discussing regression problems
	* AI 
		* Passing a test or winning a game
		* Super human performance given
			* A well defined problem
			* A loss function
			* enough data
		* Just how remarkable territory this actually covers
		* What's the loss function for more profound things like
			* criminal justice
			* couples therapy 
			* Art
			* optimal hiring
	* This is _not_ just a human issue
		* Neurophilospher 
			* Success of ANN's, notwithstanding, is a far cry from what intelligent bodies on this planet can do
		* Motivational tribes are messy - paper in 2016 [Paper]()

* Ecoli
	* bacteria have a 1 bit output
		* can go forward / back / turn like an RC car
		* trjectory os ecoli looks (see slides)
	* Energy is a function of their consumption and subsequent output
	* Is this consumption methodology optimal?
		* What actually has been optimized by evolution in this process?
		* Inverse Reinforcement Learning
			* well studied but ill-conditioned
		* Modeling a signalling and sensoring function of the bacteria
	* Genome is the reward map here
		* through this evolutionary approach we can see chemotaxis(SP?)
	* The error bars on this example are relaly big
		* colonies have lots of reward maps
		* huge variety of reward maps that do work (see slides)
	* What persists, exists
	* evolution decides on what is good or bad
	* this is not exactly optimization**..

	* Simple GAN 
	[Paper]()

		* All points are stable in wasserstein gan's
		* The combined GAN is not doing gradient descent, locally each actor here is going gradient descent of its own well-defined cost function.
			* Put together, the combined system

	* Loss functions and gradient wrt special and general relativity
		* Every actor is curving the space and this leads to general relativity
	* Many "solutions" in this bacteria case
		* signally begets collectivity
	* Optimization is not really how life works
		* it's also not how brains work

* Backprop
	* Looking at real neurons are a hell of  alot more complicated
	* Brains don't just evaluate a function
		* They develop
		* imprint
		* pre-programmed tasks
		* self-modifying
			* Looking at learning in ML, we're trying to minimize a loss by picking a particular set of weights
		* Chain rule for all of this stuff
			* backprop through a linear layer, we can see that these backprop equations look very similar to forward prop
				* there's a symmetry here
				* Also this weight update equation looks kind of Hebbian
			* ANN's generally only implement the top part of the equation
			* If we didn't do the bottom parts of the equation, it wouldn't be doing much
	* The learning part of ML is a lot more complex than the feedforward linear layers in RELU
		* There's always feedback
		* There's always temporal dynamics
		* Also
			* momentum
			* mini-batch
			* adam
			* structured ranom init
	* Neurons have all the building blocks
		* Per-cell state
		* Per-synapse state
		* Temporal averaging
		* Random number sources
		* multiple timescales
	* Can we learn to learn with these building blocks?
	* A more general, biological symapse update rule that doesn't require gradient descent
		* LSTM at every synapse
		* Shared weights, but individual state
		* Noise gate g
		* (Anti-)Hebbian
		* Neurons can behave the same way
			* Per cell state and learned behaviour for how to propagate
		* Equations are factorial and not scalary
			* Chemical activity
			* allows multiple timescales
			* mu parameters allow mixing at different time scales
			* Slow timescales needed for learning, but also useful for time- qquestions
		* Weights (or connectum)
			* learning
			* Development
		* LSTM parameters
	* In supervised learning paradigm:
		* short brain lifetimes

* Self-organizing neural cellular automata
	* Self-training NN's that are training each cell
		* reproduces a pattern
		* learns how to do this via purely local interactions

* These kinds of fundamentally social concepts and ensembles of "things" come together and create the systems we have today
	* how we find this dance

* Grand Challenges
	* Brains with fully evolved architectures
	* Understanding and characterizing evolved systems
		* realm of anthropology and sociology
	* Problem solving by artificial societies 
	* Large-scale meta learning in the Federated setting
	* Purposive "artificial ecology" engineering
	* Dynamical systems theory for neural ensembles
	* Can we define quantitative "SOTAs" for sociality?
	* Can we think about what it would mean to approach this kind of "curved space" AI ethics

* Artificial Life approaches?	


## DualDICE 

[Slides]()
[Poster]()

## From System 1 Deep Learning to System 2 Deep Learning

* These things are linked together in really interesting ways and he's going to convince us of this
* Connected to the notion of agency
* Some people think that it might be enough to take what we have and grow our datasets and computer speed and all of a sudden we have intelligence
	* Narrow AI - machines need much more data to learn a new task
	* Sample efficiency
	* Human provided labels
		* These dont catch changes in distribution, etc.
	* Next step completely different from deep learning?
		* Do we need to take a step back to classical eras?
* Thinking fast and slow
	* 2 tasks
		1. kinds of things we do inuitively and consciously and we can't explain verbally
			* This is currently what DL is good at
		2. Slow and logical, sequential, conscious, linguistic, planning, reasoning
			* Future DL
	* We're generalizing in a more powerful and conscious way in a way that we can explain
		* The kinds of things we do with system programming
* Missing to extend DL to reach human level AI
	* out of distribution generalization and transfer
	* Higher level cognitive system
	* 	High level semantic representations
		* corresponding to the kinds of concepts we link back to language
	* Causality
		* Many of these things tend ot be causal in effect
	* Agent perspective
		* Better world models
		* Causality
		* Knowledge-seeking
	* There are questions between all of these 3 things listed above
		* If we make progress in one, we can make progress in another
* Consciousness
	* Roadmap for priors for empowering system 2
	* ML Goals : handle changes in dsistribution
	* System 2 basics : attention & consciousness
		* Cognitive neurscience to understand the human side of the consciousness
	* Consciousness Prioer : sparse factor graphs
		* We can think of these things as assumptions of the world
			* The joint distribution betwen these high level concepts can be thought of as a factor graph
	* Theoretical framework
		* meta-learning
		* localized changes hypothesis -> causal discovery
			* Localized in some abstract space
	* Compositional DL architectures
		* Architectures we should explore to introduce the compositionality that we'll need to explore
			* NN's that operate on sets of objects, and not just vectors
			* Dynamical recombination
* Changes in distribution (from IID to OOD)
	* Artifically shuffle the data the achieve that?
		* Natures does not shuffle the data, we shouldn't
			* IRM paper from Bottou
	* Out of distribution generalization and transfer
		* No free lunch : need new assumptions to change this IID assumption
		* If we discard this IID assumption, we need to replace it by something else
		* Bengio posits priors might be the way to do this.
* OOD Generalization
	* The phenomenon of learner being able to genearlaize in some way to a different distribution
		* If we are a learning agent (agent = actions)
			* we almost always face non-stationarities
				* Due to actions of self (agent)
				* actions of other agents
				* movement through time and space
		* Once we start looking at multi-agent systems, it gets even more complicated
		* THERE IS NO STATIONARITY IN OUR ABILITY TO SAMPLE REALITY IN THE WAY THAT WE DO
* Compositionality helps IID and OOD to generalize
	* Introducting more forms of compositionality allows us to learn from some finite set of combinations about a much larger set of combinations that are NOT in the set of the data that we have today
	* Distributed representations
		* Helps us see why we get an exponential advantage
			* If we make the right assumptions, these things can be explained by variables and factors, and once we train a bunch of eatures we can generalize to new combinations of these features
		* Each layer is composed for the next one, and this gives us another exponential advantage
			* The one we know best we find in language
				* We call this systemasticity
	* This opens the door to better powers of analogies and abstract reasoning
* Systematic generalization
	* Dynamically recombining existing concepts into new concepts
	* Even when new combinations have 0 probability under training distributions
		* eg. science fiction scenarios
		* eg. Driving in an unknown city
	* Not very successful with the use of DL systems
		* Current methods when asking models to answer questions not in the distribution they do not know how to answer them
* Constrast with Symbolic AI Programs
	* Avoiding the pitfalls of classical AI rule-based symbol-manipulation
	* Need efficient large scale learning
	* need semantic grounding
	* need distributed representations for generalization
	* efficient = trained search (also system 1)
	* Need uncertainty handling
	* But want
		* systematic generalization
		* factorizing knowledge into small exchangable pieces
		* manipulating variables, instances, references & indirection

* System 2
	* Consciousness and attention
		* Focus on one or a few elements at a time
			* when translating we focus on a specific word to do the translation
		* Content-based soft attnetion is convenient, can backprop to _learn where to attend_
			* Soft-max that conditions on each of the elements and we can see how well we match on context
			* Attention is parallel in that we compute a score for each and decide which ones where we want to put attnetion
		* Attention should be thought of as the internal action
			* needs a **learned attention policy**
	* SoTA language models all rely on attention
		* How attention, connected to memory, can also unlock the problem of vanishing gradients
		* Operating on unbounded sets of key value pairs
	* We can think of attention as creating a dynamic connection between layers
		* as opposed to being hard coded today
		* This is great, but from the point of view of the receiving model, it receives a value but it has no idea of where it's coming from
			* We condition to the value, we have the concept of a key, an identifier for where this value came from
				* We use this as a routing mechanism
		* Downstream computation can know what the value it's receiving and where it's coming from
			* Creating a name for these objects through a form of indirection
			* we have systems of operating on sets
* From **attention** to **consciouness**
	* C-word is a bit taboo -- but maybe not anymore
	* number of theories are related to the global workspace theory
		* This theory says that what is going on with consciousness, there is a bottleneck of information in that some elements of what is computed in your brain is selected and then broadcast to the rest of the brain and influencing it
		* Conditions heavily on perception and action
			* Also gives rise naturally to the system 2 abilities above

* Relation to ML?
	* ML can be used to help brain scientist understand consciouness
	* Work in neuroscience is based on fairly qualitative defns
		* ML can help us to quantify what this actually means
	* Feedback loops help provide specific tests that we can use to measure these concepts
	* One of these motivations it to get rid of fuzziness and magic that surrounds consciousness
	* Provide advantages to these particular form of agents

* Thoughts, Consciousness, Language
	* There is as trong ling between thoughts and language, in that one can be translated between mediums farily easily though a lot of information si dropped on the floor during decoding
	* We want to explore things like Grounded Language Learning, by learning through environment interaction and perception
		* Allows a learning to get to patterns through to it's understanding of how the world works

* The consciousness prior : sparse factor graphs
	* We can use these systems to encourage our learning systems to do a good job at out of distribution reasoning
	* Sparse factor graph
		* attention : to form conscious state, thought
		* A thought is low dimensional object
			* We sample these from a larger higher dimensional conscious state
			* The conscious states that we sample 
			* The thoughts we consciously have are pushed through this consciousness bottleneck
	* What do these computations mean
		* Some kind of inference is required
		* What kind of joint distribution of high level concepts are we reasoning about
	* Think about the kind of statements we make with natural language
		* "If I drop the ball, it will fall on the ground"
			* true but involved very few variables in that the statement only contains a finite number of words
		* The relationship that I need to descibe can tightly capture the elements of this joint probability through very few variables
	* Disentangled factors != marginally independent, eg. hand & ball
		* We think of these as having this very structured joint distribution
		* They come with very strong and powerful relationships
		* Instead of imposing a very strong prior of complete margin independence we can find some prior that finds a joint distribution between these high level variables
* Meta-Learning : End to end
	* Meta-learning is learning to learn
		* Backprop through inner loop 
		* Having multiple timescales of learning 
			* iterative optimization like computation
			* out of loop evolution
		* In this way we can talk about evolution algorithms, etc. and when we talk about this in the life of an individual 
			* lifetime learning is the outer loop and local interaction through time is the inner loop
	* We can train it's slow timescale meta parameters to generalize to new environments
	* What kind of hypothesis can we make?
		* becaue these actions are localized in space and time, because these things are locallly temporal then we can try to understand
		* Independent of cause and mechainsim -- from an information perspective
			* learning from one mechanism tells you nothing of the others
			* if something like this changes due to an intervention then ew only need to adapt the portion of the model that has to deal with that part of the distribution
			* It can be explained by a tiny change
	* Good representations of variables and mechanisms + localized change hypothesis
		* few bits needed to adapt to what has happened

* How to factorize a joint distribution in this way?
	* Learning whether A causes B
		* Learner doesn't know but we might observe just X and Y
		* Turns out, if we have the right composition then we can use this to learn about how to map X to Y, such as things like pixels that don't have causal structure in that image itself
	* The assumption that these high level variables are causal doesn't work on pixels
		* We cna't find a pixel that causes another pixel
	* Learning neural causal models
		* we can avoid the exponential number of grpahs that need to be considered through this
		* ONe of the things found was that in order ot facilitate the causal structure the learner should try to infer the intervention on which variable it was performed
		* most of the time our brain is trying to figure out "What caused the changes that I am seeing?"
	* Able to find these commonly used causal induction methods
		* Attacking this problem in a deep learning friendly way
			* defined obejective with some regularization

* Operating on sets of objects
	* Using dynamically recombinations of objects
	* Recurrent Independent Mechanisms
		* operating on sets of these objects
		* applied to recurrentness
			* state is broken into pieces
			* constaining the way these networks are talking to eachother in that they done in sparse and dynamic ways
			* vectors aren't the standard vectors but rather sets of pairs 
				* networks are exchanging variables along with their type (key, value) pairs
					* leads to better out of distribution generalization than those that don't use these structures
	* Tested in reinforcement learning
		* found it helped in atari games

* Recap
	* Conscious processing by agents, systematic generalization
		* Sparse factor graphs in space of high-level semantic variables
		* Semantic variables are causal : agents, intentios, controllable objects
	* Shares "rules" or modules that are reusable across tuples
		* A particular subnetwork recieves input that is different dependent on context
			* This can be applied to different instances in that it's much more like a bayes net but the same parameters can be used in many spaces
	* another really important hypothesis is that the changes in dsitribution are mostly localized if we're presented information
	* Things preserved across changes in distribution have to be grounded in that they're stable and robust to stationarity