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

Threat Model
	* Malicious Adversarys
		* Perform arbitrary inferences on the learned model to extract data points
	* Curious onlooker
		* Passively looking at updates

Central Differential Privacy with small privacy parameters (Epsilon <= 2)
	* moment accounting

Local Differential Privacy 
	* local pertubation on the device modifies the vector into a new vector z = M(w)

Control - allows users to opt into this feature within device settings
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
	* adding a public value : [x] + y = (x^(0) + y)
	* Multiplication needs triples of encrypted random numbers with the property that [a][b]=[c]
		* once we have these tiples we can then generated a share for a, b, c
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