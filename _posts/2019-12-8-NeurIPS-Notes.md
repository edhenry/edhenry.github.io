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

## Mapping Emotions: Discovering Structure in Mesoscale Electrical Brain Recordings 

[Slides]()

* Brain is an organize that integrated bio-information with electricity
	* working on integrating across discplines to get better models of brain functions

* Causality???
	* If we understand how the brain does what it does, we can reverse engineer it and use that to understand it better
	* Can we turn on and off other areas of the brain
	* We come to this conclusion around causality because human beings have observed the earht other and over again in many contexts and test these assumptions using models and test these assumptions and map them back to our models
		* This is important for neuroscience
			* Many of the manipulations of the brain cause it to not function how it does naturally
				* It might make more sense to observe the system over and over again
				* We can test these models
					* If that's the case, the human body is 98 degrees farenheit
	* Depression (DSM)
		* depressed mood
		* diminished interest
		* increase or decrease in appetite
		* hypersomnia or insomnia
	* Is MDD prevention a viable therapeutic strategy?
		* Imagine a disease case where someone has a heart attack of failure
			* make that heart pump more normally
			* Idea is to make a diseased heart and make it function more normally
			* One of the things that has the greatest impact on the system of the heart was aging
				* measured variables in lifestyle that might help predict heart attack later on in life
	* Emotions at latent networks
		* The idea is that we use fMRI to look at changes in bloodflow in the brain
			* Using these changes as a proxy for brain activation
			* Taking healthy controls, or students
			* Put them in a scanner and let them watch movies while inducing emotions
				* Use ML to see what emotional patterns were induced
			* After about 20 mins they tap the students and ask how they were feeling
			* This suggests we're able to liberate the emotion from the patients' brain without self-reporting
* Assumptions
	* Assumption 1: Emotional encoding at the second timescale
		* This is important because people have classically thought about controlling variables and repeatedly observe a system
		* Useful for studying vision, motor function, sensory systems, etc.
			* This system might be built to do something _very_ different than emotional systems
		* When we have a system to process emotions, we want something that has information resonance at a slower timescale than moving arms, and legs, etc.
		* Emotions are encoded at the timescale of seconds
	* Assmption 2 : Emergent Properties
		* We have a cell property and it traverses a neuron and checmical information is sent down the axxon
			* hard to think about emergent properties of these systems
			* a seizure is an emergent property of the brain
				* wouldn't generate this phenomenon without more than 4 cells
		* The system is working together in an integration fashion to create these properties
		* THINK SLEEP
		* LOcal field potentials can be used to measure certain properties of the system
		* LFP coherence (functional connectivity)
		* sycnhrony or coherence
			* we can infer directionality in a circuit
		* Coupling between cell firing and LFP activity
	* Assumption 3 : Local Field Potentials reflects the activity of populations of neurons (emergent features)
		* Trying to find things that generalize across brains, species, and inviduals
	* Latent Network Model
		* Each layer is useful for one of things we wanted to do
			* Each of us had to believe one layer of this model
		* Information across frequency and information that is leading or lagging
		* 6000 things that we could measure and quantify in an animals brains
		* Phase offset
	* Acquiring brain network behavior

## Agency and Automation

[Slides]()

* Hype and sensationalism drive some of the interest but there is a substory there around automation
	* Fei Fei Li - write in NYT that enthusiasm for AI is preventing us from reckoning our immersion into it...
	* Michael Jordan
		* Need well thought out interactions with humans and computers
* These are not new 
* We have a 60 year old design challenge to find an optmiality between divisions of labor and automation
	* Automation and user control
		* Waht si the appropriate balance here?
		* Challenges of automation
			* Automated methods may be biased or innacurate
			* These concequences can be quite damanging in the real world
			* Loss of critical engagement and domain expertise
		* We lack a global view as humans and over-weight local information
	* Balancing automation and control can be done through building models of capabilites, actions, and controls around the tasks that we perform
* 3 Examples
	1. Exploratory Data Visualization
		* incorporate tasks that the user is trying to achieve into the design of the visualization of the data
		* See slides for multi-verse analysis of the topics that might be present in these documents
		* What makes a visualization good?
			* Task specific and subjective references
			* Foundational issues in perception that we can build upon
		* Shows the long standing results in psychophysics in how our perceptual system can quickly decode visual types of information
		* Common exploration pitfalls
			* Overlooking data quality issues
			* Fixating on specific relationships
			* Many other biases...
		*Data Voyager
			* examples in slides
			* We want to suppoer systematic considerations of the data
			* Model user's search frontier, optimize for related chart specification, seeded by the user's current docus
			* Candidate charts pruned and ranked using a formal model of design constraints


## Bayesian Deep Learning Workshop

### Gaussian Process Behavior of Wide and Deep Neural Networks

[Slides]()

* Paper has been around for almost 2 years now
* Lots of forward work -- along with an extended paper on arxiv today along with code
* Data efficiency is a serious problem for deep RL
* Prior and weights are typically very difficult to interpret
	* Why do we expect good performance?
	* Possible what we are doing inference with a terrible prior
* Seminal results of the paper
	* 1994 - showed that a single MLP with K hidden units, with a carefully scaled prior
		* scaling outgoing weights by 1/K -- as you take the limit as k -> inf
			* the vector converged to distribution multivariate with mean 0 and unit variance
			* some form of gaussian quadrature
			* proves this with standard multivariate central limit theorem
* Central Limit Theorem
	* 1 dimenstional CLT there are some interesting things
	* Random variables converges to CDF at all points where the CDF is continuous
	* CLT says that if we consider IID rv with mean 0 and unit variance
	* Some sublties
		* Consider an IID sequence of 2 possibilities [-1, 1] with P(0.5) has mean 0 variance 1
		* We can define a set A
			* Then for all n where A has probability zero under N(0,1)
			* This set has 0 probability under this distribution
				* be careful with what convergence of distribution actually means
* What does this mean for a stochiastic process to converge in distribution
	* carefully scaling the prior
	* weights coming out of these layers will have 1/k_1 and 1/k_2 respectively
	* c_2 is a normal quadrature is defined in terms of the covariance of the previous layer making it a recursive kernel definition
* Deep Neural Networks as Gaussian Process
	* Released same day and accepted as same confernece
	* Check this paper out as well [Paper]()
* Rigorous proof provided
* Why would we expect a CLT here at all with multiple hidden layers
	* Radford Neil
		* Feeding a single data point through and we can look at the f_1 units - each will converge independent of other variables
	* Multiple input data points
		* there is a correlated normal vector at each f^(1)
			* at some point, increasingly independent vectors converge to a correlated normal vector
	* Problem with the argument
* Preliminaries
	* Need a convergent non-linearity
		* Draw a bounding envelope on any point around the non-linearity
			* we might get something that might not be defined if we don't, it effectively stabilizes everything
	* Think about the network as an infinite sequence of network
		* The hidden layers may grow at different rates as long as they all tend toward infinity
	* Formal statement of the theorem
* Proof sketch
	* proceed through the network and by induction starting to closest data
	* at each layer, reduce the problem to the convergence of any finite linear project of data and units
* Exchangability
	* An infinite sequence is exchangable if any finite permutation leaves its distribution invariant
	* de Finetti's theorem
		* an infinite sequence of random variables is exchangable iff ti's IID conditional on some random variable
* Exchangable CLT [slide]()
	* applies to triangular arrays

* Experiments in the paper are relatively small data with low dimensionality [Slide]()
	* In the majority of cases considered, the agreement is very close
	* one can't tell the difference between the GP and 3 layer NN
	* Slide shows, empirically, there seems to be little difference between a standard GP and a DNN with 3 hidden layers

* Limitations of Kernel Methods [Slide]()
	* This property might not be a good thing
	* Kernel methods are affine transformations of the training outputs
	* This limits the rperesentation that we can learn

* Deep GP's
	* Not marginal GP's because they have finite restrictions in the norm
	* This prevents the onset of the CLT

* Subsequent work [Slide]()
	* CNN's also converge to GP's
	* Neural Tangent Kernel considers not what just happens for the initial distribution of the NN, but also what happens when we apply gradient descent

### The Natural Neural Tangent Kernel

[Slides]()
[Paper]()

* Background
	* Vectrorize the output of the NN's into an n x k vector
	* we know that the application that we have done will handle 1D output
	* All theoretical results apply to multi-outputs
	* We know that NN outputs are a function of the parameters which in turn are a function of time (think evolution of gradient descent)
* Natural GD
	* appealing because it has convergence, covariance, and invariance under reparameterization
	* Fisher Information Matrix allows GD to take the curvature of the distribution space into account
		* Small changes in parameters can effect the training dynamics
		* Inverse Fisher allows us to take into account this space's information geometry
* Concatenated Fisher Information
	* we can condition the FIM on a single data point x

* Training dynamics under natural gradient descent
	* Natural Nerual Tangent Kernel includes the fisher information matrix which includes the distribition geometry into account

* Assumptions
	1. Network overparameterization
	2. Positive definiteness
	* Implications

* Computing the NTK yields an interesting result
* Bound on prediction discrepancy [Slide]()
* Empirical results
	* Symthetic data [Slide]()
	* Theoreitcal bound is meant to be tight
	* The values increase further away from data -- see tails of the plot
	* Comparing the predictive distribution -- see slides
* Future Direction
	* Approximate inference
	* scaling to larger datasets
	* Classification tasks
	* Generalization analysis

### On Estimating Epistemic Uncertainty

* There is huge interest in the intersection between Neural Networks and Bayesian Believers
* epistemic estimation is REALLY important for areas with high risk 
	* bayesian or not there is a really great potential to healthcare and autonomous driving
* Type of Uncertainty [Slides]()
	* Epistemic uncertainty "how much do I believe this coin is fair?"
		* models' belief after seeing the population
		* reduces when we have more data
	* Aleatoric Uncertainty - "What's the nest coin flip outcome?"
		* Individual experiment outcome
		* non-reducible
	* Distribution Shift - "Am I still flipping the same coin?"
		* Indicating a change of the underlying quantity of interesting

* Quick intro to BNN's [Slide]()
	* instead of learning point updates, let's put a distribution in place here over the parameters
	* in practice $p(w|D)$ is intractable
		* Find an approximation q(W) \approx P(W|D)$
	
* Weight space uncertainty is less interesting
	* in many cases NN's weights are NOT scientific parameters we're interested in
	* symmetries/invariance in parameterization
		* exmaples like swapping nodes and scaling of weights, we're still approximating the same function

* This introduces vagueness [Slide]()
	* sample weights from the Q distribution
	* folklore belief for function-space (or output-space) uncertainty:
	* "Epistemic uncertainty should be high when new input is less similar to observed inputs"
	* **What do "high uncertainty" and "less similar" mean qualitatively?**
		* This is typically "eye-balled", leaving it to be subjective by definition
		* There is really no agreeable diescription of where and by how much it should be higher
* Evaluating by comparing to references [Slide]()
	* BNN's performance relies on a approximate posterior
	* Evaluating inference:
		* computes some distance metric between q(W) and p(W|D)
	* Function space "reference posterior" for BNN regression:
		* some hope in function space
		* wide BNN has GP limit (under certain conditions)
		* for regression problems $p_{GP}(f|D)$ is tractable

### The big problem with meta-learning and how bayesians can fix it

[Slides]()

* How do we accomplish learning from scratch or from _very_ small amount of data
	* Modeling image formation
		* geometry of the image
		* SIFT features, HOG features, etc.
		* Fine tuning from ImageNet features
		* Domain adaptation from other painters
	* Fewer human priors as we move down the list above
* Can we explicitly **learn priors from previous experience**?

* Brief Overview
	* Given 1 example of 5 classes :
		* classify new examples
* How does meta-learning work?
	* One approach : parameterize learner by a neural network
	* Another approach : embed optimization into the learning process
	* The Bayesian perspective : learn priors of a Bayesian model that we can use for posterior inference

* The problem 
	* How we construct tasks 
	* What if label order is consistent?
		* A single functional can solve all the tasks
		* The network can simply learn to classify inputs, irrespective of the data distribution

* Meta-training to "close the box"
	* If you tell the robot the task goal, the robot can **ignore** the trials
	* another example : pose estimation and object position
		* memorize the post and orientation of the meta-training set
		* at meta-test time, without knowing the canonical orientation, we don't be able to accurate predict the orientation

* What can we do about this?
	* If we had a proper bayesiaan meta-learning algorithm that was learning a proper posterior, we might not have this problem
	* However, I'm not sure if we have the tools to create a proper meta-learning algorithm
	* If the tasks are mutually excluse, a single function cannot solve all the tasks (due to label shufflinf, etc.)
	* If tasks are _non-mutually exclusive_, a single function can solve all tasks
		* multiple solutions to the meta-learning problem

* Meta-regularization
	* Control the information flow such that we can do zero-shot learning from the data
		* minimize the meta-training loss and the information contained within the parameters of the model
		* regularizing the weights forces the model to use information from the data as opposed
	* Can combine this with a favorite meta-learning algorithm

* Does meta-regularization lead to better generalization?
	* arbitrary distribution over $\theta$ that doesn't depend on the meta-training data

* Meta-world benchmark

### High Dimensional Bayesian optimization using low-dimensional feature spaces

[Slides]()

* Motivation
	* Experimental design problems that can be cast a a global optimization over some parameter space
	* optimizing on non-linear projections

### Function Space Priors in Bayesian Deep Learning
	* WHy do we care about function space priors?
	* Lots of testing methods for bayesian approaches
		* see slides
	* But these all have non-Bayesian approaches that are competitive 
	* Three X's
		* Exploration
		* Explanation
		* Extrapolation
		* These cases all depend crucially on having good priors that reflect thr structure of the underlying distribution
	* Compositional GP Kernels
		* GP's are distributions over functions parameterized by kernels.
		* Primitive Kernels	
		* Composite kernels
			* taking products of kernels
				* This can express things like periodic structure that gradually changes over time
			* No need to specify structure in advanced and can be inferred online during training
* Structured Priors and Deep Learning
	* Demonstrates the power and flexibility of function space priors
	* Problems
		* Requires a discrete search over the space of kernels for each candidate structure
		* Need to re-fit the kernel hyperparameters
* Differentiable Compositional Kernel Learning for Gaussian Processes
	* Neural Kernel Network
		* represents a kernel
		* inputs are 2 input locations
		* output is the value between them
	
### Try depth instead of weight correlations

* Challenge the assumption
	* ASsumptions that the approximate posterior that we use to model our BNN, ought to have correlations between the weights
	* Mean-field assumption that our weight distributions are independent of eachother because we're avoiding intractability
	* This is less true as our neural network gets much deeper

* Why might our approximate posterior need to have correlation between weights?
	* Maybe the true posterior does?
	* A lot of intuitions we have come from this small interpretable single layer 4 neuron model
		* What we think is that alot of these effects disappear as we get deeper and deeper networks

* With depth, we can induce rich correlation over our output distribution with mean-field weights
	* one way to do this is to have covariance between $\theta_{1}$ and $\theta_{2}$
	* As we get a deep network, we can get richer covariance structures
	* 2 inputs and 2 outputs with a simple weight layer w
		* assuming linearity
		* mean-field approximation
	* Lesson from the linear case
		* 3+ mean-field layers can approximate one full-covariance layer
		* More layers allow a richer approximation
	* Measuring the price of this mean-field approximation in NN's that _do_ have non-linearities
		* HMC true posterior
		* fit a full-covariance gaussian
		* fit a diagonal covariance gaussian
		* measure the difference between them, and it should give us an understanding of how costly the extra assumption of diagonality is
	* Measuring the 'price' of the mean-field approximation [Slide]()
		* hold parameters model throughout this testing
* What are the implications here?
	* Rely less on UCI evaluation with a single hidden layer
	* More research into **other** problems with Mean-Field Variational Inference
		* E.g. sampling properties of high-dimensional gaussian ("Radial BNN's")
	* Less research into **structured covariance variational inference**



## Inset workshop name

### Intuitive Physics for Robotic Manipulation of Liquids

[Slides]()
[Poster]()

* Interaction with liquids happens every day
	* Specific containers and specialised tools to manipulate these liquids
	* We can approximate the way these things will behave
	* The shape of the continaer has causal influence over the way that liquids interact with it
	* Viscosity of the liquid has intersting causal properties
* Thinking about this from a robotics perspective
	* Some of the things very natural to us are hard for robots
		* the complex properties of liquids makes this hard for robots
* What is it that we, as humans, do to help this manipulation
	* CogSci theories
		1. We have some approx simulation in our heads that enable these predictions
		2. People have shown that we can invert this simulator in our head and make predictions about properties in our heads
		3. Different types of interactions can give us different cues about viscosity
	* We need some sort fo fast approxiate like thsi for embedding in robots
	* We don't need exteme accuracy but rather representing these objects in a more approximate and efficient way
* Intuition as approcimate simluation
	* NVIDIA Flex
		* Position based dynamics 
		* As with any simulation we use, we have a reality gap
			* This is discrepancy between observation in the real world and simulation environments
		* Sources of error
			* model approximation -- not much we can do here
			* parameterization -- we have to set the parameters of the model and without correct settings we'll get variance in our predictions
		* Sim2real discrepancy is what we're trying to track in this
* Two stage process
	1. Estimate parameters & learn to pour
		* we want generative models to enable adaptatoin to dynamics of the environment
		* even though it's the same experiment, we want to minimize spillage, but at the same time we want it to spill a bit because it's informative
* Learning how to pour
	* How to use the sim here
		* action and observation spaces are as follows
		* count the number of particles that fall outside of the container
		* measure the spillage with a scale, and noramlize such that we have a % spillage to compare between the two domains
		* Find the relative distance between source and target container while measuring how fast it's being filled
	* The way we do this is to model this as a Gaussian Process
		* pour N times (37 in paper though 15 should be enough)
		* learning combinations of velocity and relative spillage
		* after learning this we transfer it right to the robot
	* Approximate fluid simulation is useful
		* geometry of the container causes high spillage!
		* initialization of policy with simulation works best
* To stir or not to stir
	* calibration to properties of the liquid through perception
	* The way we calibrate is to perform, in a synchronous way
	* Cohesion models the best the change in viscosity in the real world
		* the condition is the thing that is being modeled
		* simulator cannot model adhesion
			* these characteristics cannot be simulated
			* is there a way we can model this friction coefficient that might be present in specific liquids?

### Perception and action from generative models of physics

[Slides]()

* Goal of this research is to study generative models of physics from a cognitive science perpsective
	* Hemholtzian idea of perception as inverse optics
	* Some uderlying true state of the world
		* but we don't have access ot that
		* we only have access to retinal images
		* there is some lawful set there
			* can we invert this image, knowing what we know from optics, to derive information about the world
	* The world changes over time and gives rise to a sequence of images that we see
		* these changes happen in lawful ways such as dynamics
			* We don't want to treat these observations as IID
			* We can use this to constrain our inferences
			* Perception is constrained by dynamics
		* People's judgements about the slant of a ramp given the visual state of the ramp
			* as our perception of the ramp changes, if affects how we perceive the world
* What are these dynamics in the world and how do we capture that?
	* Intuitive Physics ENgine
		* The generative models we have in our heads are based on object baed representation
			* Some probability distribution presents a range of world state
			* This gives us a range of possible ways the world might unfold
				* We run out model forward and count up the number of blocks
		* Important features
			* object based
				* shows object based importance in early human development
			* probabilistic model
				* Not just one possible future, but  range that we can make predictions over
			* We don't need a veridical model of physics but one good enough to action plan
			* This physics engine should favor speed and efficiency over precision
			* MOdel is generalizable in that we don't need to learn separate physics models for all situations
		* How do we do this?
			* Predict - have a generative model of the world, ask what happens next, run out the model and observe
			* Probabilistic framework unlocks a lot of additional capabilities for perception
			* Perceive causality -- remove A from simulation
			* Make plans and choose actions based on these model run outs
* Physics in the loop of perception
	* Perceiving what is in the world
		* seeing occluded objects [Papers]()
		* seeing surprising events [Papers]()
	* Understanding actions in the world
* Seeing Occluded objects with generative models in the loop
	* If we have a set of objects that have cloth draped over them, we can infer what object might be under that cover
	* We need some sort of generative model that allows us to internally ask "what would this look like with a cloth over it?"
	* See slides for how to model this occlusion phenomenon
		* use dynamics and physics of cloth to find a draped cloth geometry
		* Inference with Bayesian Optimization is key here
		* Understanding _How_ that cloth might drape is important for understanding what something occluded with cloth might look like
* Seeing Surpriving Events like an infant
	* Detecting violations of expected dynamics
	* Permenance
		* objects can't teleport
	* Solidity
	* COntinuity
		* when objects violate these properties of how objects work, then they can update their model of the world
* What do we need to build into an agent such that it can percieve the world but then update their understanding of the world according to some surprise factor
	* Perceive violations of these principle drives learning

* ADEPT Model [Slide]()
	* Given an image - we first extract object information
		* approximate de-renderer
		* this object has attributes has understanding of position, velocity, etc.
			* shape information is thrown away
	* propose object masks
		* feed through renderer gets object properties
	* Internal scene representation -> physics observations
		* objects are moving at certain velocity and objects interact, they don't move through eachother
	* We want to match the above observations against a "ground truth"
		* this isn't matching in pixel space, but rather matching wrt objects
		* we also have to gracefully deal with unexpected events
			* disappears and we want to handle it by saying this is something weird that happened, but this is my new normal and I no longer need to track it
	* Measuring Surprise
		* violation of expectations (from psychology)
		* creation of a bunch of physics based violation types
			* these match to infant understanding principles
* Infants don't see non-physical events
	* this allows us to constrain our space of potential evaluation
	* objects in shapenet
* Alternate Theory 
	* Bootstrap these princples above
	* Can we learn this from enough data?

* Rapid trial and error (repuposing of objects)
	* example of a stake and a tent
		* rule out branch, pinecone
		* pick rock
	* finding representations of the properties of these objects is inherent to planning in these situations
	* this seems to be a core feature for people
	* PHYRE benchmark
		* Focused on model-free RL from balanced datasets
		* learn generative model of the dynamics of the envornment
	* Visual foresight for learning to push objects with tools
		* from vision required many samples + demonstrations
* SSUP Framework
	* sample, simulate, update
		1. Prior
		2. Internal simulator
		3. Learning mechanism
* Conclusions
	* Causal models of dynamics are important for perception and action
	* Types of representations & dynamics are crucial
		* object-based, approximate world models
	* Just generative models is not enough -- requires additional information

### 

* Neural Netwoks and CONVNETS are super dense
	* we're grabbing much of background context, etc that don't necessarily matter
* Hierarchical compositionality
	* Way fewer parameters
		* the whole model was quite interpretable ane debuggable
	* each unit was a node in a graph -- allowing representations of images in graphs
	* inference was done in a very hacky way

* AI Today
	* AI for simulation
	* Simulation needs a lot more learning involved
	* Open Problems
		* 3D Envornments / Scenes
		* 3D Objects
		* Activities
		* Behavior
	* Scalability, realism, diversity : Learn how to simulate!
	* Scene composition
		* making this a little more scalable
		* In gaming, worlds are build using sort of probabilistic 
* Meta-SIM

### Learning spacial invariant object properties

* Cross-bite challenge 
	* Building Machine That Learn and Think Like People
* Unsupervised Object Tracking
	* Training (no annotations!)
		* find donstruction of videos in terms of moving objects
	* Testing
		* new set of images from the same distribution and eval performance same as supervised learning
	
	* Sequential Attend, Infer, Repeat (SQAIR) [Paper]()
		* Unsupervised object tracking 
			* Variational autoencoder
			* trained by maximizing the ELBO
				* hopes to learn the dynamics of the objects
	* Spatially Invariant, Label-Free Object Tracking (SILOT)
		* new architecture
		* includes features to help it scale up
		* allows objects to condition and coordinate on eachother
			* we can sidestep the require sequential structure
