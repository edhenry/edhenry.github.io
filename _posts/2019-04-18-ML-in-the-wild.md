
# Machine Learning in the Wild

I won't be covering anything purely technical in this post, but I wanted to get some thoughts out there on what it takes to bring Machine Learning into production,

"Any sufficiently advanced technology is indistinguishable from magic." - Arthur C. Clarke

This quote rings true today as much as the day that it was written. Especially when it comes to Machine Learning. I work with customers every day who are curious what this new magic called Machine Learning(ML) is and how they can apply it to a problem they're facing. There's no denying that ML is something that we should all be paying attention to but as we start to wade into the waters of ML, and discussing the process of framing the problem, testing our hypotheses, and bringing a system through to production we need to understand that this requires a bit of a shift in our traditional engineering process. In this post we will outline the general process that I've followed while helping customers move from idea/MVP to production.

## Don't start with Machine Learning

ML in undoubtedly changing the way we approach solving problems in the world today. More importantly, however, is knowing what problem it is that you're trying to solve. Like many sciences and engineering discplines practitioners are looking to understand a problem domain in an effort to guide decision making processes. It's this decision making process that is what we think about when approaching a problem. I want to highlight a few key questions one should ask about the problem at hand before rummaging through their toolboxes for an overpowered approach. Some of these key questions are as follows.

* What decision are we trying to make?
* Is there an existing decision making mechanism in place?
* Are we trying to replace this decision making mechanism?
* Who is this decision impacting?
* What will the potential outcomes of this decision be?
* What properties of the system am I trying to capture?
* Do we have domain experts involved?
* Do we have the _right_ domain experts invoved?

None of the questions I've provided have anything to do with Machine Learning directly. Rather, they're asked to help guide the process of applying ML and measuring it's utility in a given problem domain. These types of questions will help you scope the problem you're working on and provide a certain acceptance criteria.

## Data and the system generating it

Once we have a thorough understanding of the problem that we're trying to apply ML to, and we understand what decision it is that we're trying to make / replace, we can start to investigate what data that system generates. Many of the customers I've worked with think about this process as _just_ the ML oriented questions around the types of features, what types of random variables they are, what classes we may want to define for a supervised learning problem, etc. We also need to understand questions not directly related to the data generating system, itself. Questions around data collection, storage, access, governance, provenance, etc. Questions like :

* How is the data being collected?
* What steps are being taken to validate the data collection mechanisms?
* How reliable are the measurements being collected?
* Where is the data being persisted?
* What controls are in place to manage access to this data?
* How is this data being versioned?

These questions will influence the success of your ML project, overall. Without data, we're left without the ability to use this powerful new technology.

## Deliverying Production Grade Software

The next step in the adoption of ML is understanding not just what software is, how to write it, and how to use an SCM system to manage releases, but one also needs to understand what running software in production entails. Anyone who has been part of the industry for any length of time in the last 5 to 7 years has heard the term DevOps before. This term is just a blend of the terms Development and Operations and the concept of blending the two words captures the idea of minimizing the distance between the two general concepts. What it means to be a developer on a production level system requires a thorough understanding of not only the code base, but the properties and characteristics of our production grade systems and what the upper and lower bounds are on the performance of our systems.

You may have also heard the term **machine learning engineer** used within industry, as well. This new title and role is a result of the differences, yet seemingly similar, approaches needed to move from experimentation to production in ML systems. Some of the things to think about involve both the data acquisition, and machine learning (training and serving) pipelines.

* What does testing look like for our pipelines, end to end?
* How do we perform validation of performance of our system (not just the model)?
* How can we provide tooling to measure for concept drift?
* How can we qualify acceptable throughput of our system, end to end?

## Now for Machine Learning

Now that we have a thorough understanding of the problem domain, use case, data generating system, and all of the software required to stand a system up in production we can start to loop back to the opening thoughts of this post. We need to understand what methods we might be able to use to describe the phenomenon that we're trying to capture. Typically in Machine Learning by describe, we mean capturing the variance of the overall system in a way that we can use to ultimately make decisions.

In order to understand how we can apply ML, in production, we need to understand everything outlined above because it will impact the choices we make from a modeling perspective. Anecdotally we don't want to make the decision of using a non-parametric model in a given problem domain where the datasets might be growing rapidly. A concrete example of a use case that I've run into in this effect has been modeling data networking traffic characteristics. The rate at which data is generated and collected depends on the number of devices sending information over a network. With the types of connectivity we see today this can quickly become a show stopper from moving a model into production, especially if we have to layer feature engineering into the pipeline, on the fly, as well. A non-parametric model will quickly run out of resources in terms of computational and/or time, depending on the problem domain.

Understanding the use case, the decisions we're trying to make, what mechanisms are in place to collect and persist the data that our system is generating, and the software tooling and ecosystems required to put pipelines into production are imperative to providing a seamless experience of applying Machine Learning in the real world.

## End-to-end Throughput

Last but not least after all of the steps outlined above are thoroughly understood we can start to provide the engineering guarantees we're used to in the technology world. Anyone who has heard the term speeds and feeds understand that we're talking about the end-to-end throughput of an entire tech stack, from start to finish. Once we are able to effectively measure all of the components of our ML system we can then identify bottle necks and work to alleviate them in a measureable and defensible way.

Overall, deploying Machine Learning in production may start out as feeling orthogonal to the traditional software engineering methodologies, but once an organization works through a few proof of concept or minimum viable product efforts, they will then understand what it takes to provide a meaningful end-to-end experience for using this new magic called Machine Learning.
   



The most important thought to keep always in the back of your mind when approaching a problem, using data, is that ML _**should not**_ be the first technology you reach for. Machine Learning typically consists of either modeling an input space, or set of all possible input values of a _**data generating system**_, or modeling properties of the underlying _**data generating distribution**_ of the data generating system. In this, exists an implicit assumption that we understand the data generating system, and problem domain that that system exists in, enough to describe the system in a simplified way. This is what an ML model is. It's a simplification of reality; reality being some form of physical process that occurs in the natural world.

The process of ML is, generally, hypothesis driven in that we have some data set that is representative of our underlying data generating system and/or distribution, and we would like to capture the variance contained within that data set to describe the possible state space that this data generating system has existed in, in the past. Through this process we then use these specific data points to draw more general conclusions about that system, using current more current. This process is called [inductive inference](https://en.wikipedia.org/wiki/Inductive_reasoning). The nature of this hypothesis driven approach is orthogonal to traditional engineering outfits in a few key areas. 

The first is an inherent understanding of the complexity contained within the algorithms and data structures we use to manipulate the data types that we have that exist today. To make this concrete, if I were to attempt to implement a traversal algorithm over a binary tree data structure, I have a concrete understanding of the data types, data structures and algorithms that I will be using, along with a concrete understanding of the computational complexity of the algorithms that we'll be implementing. Machine Learning inverts this process. We have an understanding of our data types that we'll be evaluating are generally well understood. A few concrete examples are pixel intensity values for images, raw audio waveforms for use in speech synthesis, ip addresses and port numbers for network data. What we don't have a concrete understanding of is the computational complexity with respect to evaluating these data types. Writing an algorithm that could account for all combinations of pixel values for all photographs of all animals in the world is intractible. This drives the hypothesis driven approach. We're going to investigate and model our data set in hopes of uncovering the hidden complexity contained within our data.

To summarize, traditional software engineering houses the complexity of computation within the algorithms and data structures themselves. In Machine Learning, we're attempting to discover this complexity with no formal guarantees that our data set is representative enough of the system we're trying to model...ergo hypothesis driven investigation.

## 
