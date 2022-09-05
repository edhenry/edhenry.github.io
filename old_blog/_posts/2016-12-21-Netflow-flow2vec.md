---
layout:     post
title:      Netflow and word2vec -> flow2vec
author:     Ed Henry
tags: [Python, Machine Learning, word2vec, netflow]
---

```python
import pandas as pd
import numpy as np
#import pyhash
import gensim
import multiprocessing as mp
from joblib import Parallel, delayed
import concurrent.futures
from pprint import pprint
import random
import mpld3
import re
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import manifold
from sklearn.decomposition import PCA, TruncatedSVD

%matplotlib inline

# Enable mpld3 for notebook
mpld3.enable_notebook()

# Instantiate hasher object
#hasher = pyhash.city_64()

# Method to strip white test
def strip(text):
    return text.strip()

# Method to set dataframe entries to integers
def make_int(text):
    return int(text.strip(''))    

# Method to match IP against flow srcIP
def sort_ip_flow(ip):
    # List to house flows when matches
    flows_list = []
    # Iterate over tcp_flows list
    for flow in tcp_flows:   
        # Comparison logic - flow[1][3] corresponds to SrcIP in flow tuple
        if ip == flow[1][3]:        
            # Append match to flows_list
            flows_list.append(flow)
    # Return dictionary of IPs and flows
    return {ip: flows_list}

def process_flow(flow):    
    # Create hash of protocol
    proto_hash = hasher(flow[1][2])        
    # Create hash of SrcIP
    srcip_hash = hasher(flow[1][3])        
    # Create hash of Sport
    srcprt_hash = hasher(flow[1][4]) 
    # Create hash of DstIP
    dstip_hash = hasher(flow[1][6])    
    # Create hash of Dport
    dstprt_hash = hasher(flow[1][7]) 
    # Cast flow entry as list for manipulation
    flow_list = list(flow)       
    # Insert hashes as entry in tuple for each flow
    flow_list.insert(4, (str(proto_hash), str(srcip_hash), str(srcprt_hash), 
                         str(dstip_hash), str(dstprt_hash)))    
    # Re-cast flow entry as tuple w/ added hash tuple
    flow = tuple(flow_list)
    return(flow)

def single_hash(flow):
    flow_hash = hasher(flow)
    flow_list = list(flow)
    flow_list.insert(4, str(flow_hash))
    flow = tuple(flow_list) 
    return(flow)
    
```


```python
# Import netflow capture file(s)

flowdata = pd.DataFrame()

cap_files = ["capture20110810.binetflow","capture20110811.binetflow"]

for f in cap_files:
    frame = pd.read_csv(f, sep=',', header=0)
    flowdata = flowdata.append(frame, ignore_index=True)

# Strip whitespace
flowdata.rename(columns=lambda x: x.strip(), inplace = True)
```


```python
subsample_cats = flowdata.loc[:,['Proto', 'SrcAddr', 'DstAddr', 'Dport']]
subsample_labels = flowdata.loc[:,['Label']]

subsample_cats_1 = flowdata.loc[:,['Proto', 'SrcAddr', 'DstAddr', 'Dport', 'Label']]
```

## Flow2vec - co-occurence idea for flow data

Attempting to find some co-occurence patterns in the flow data according to how an algorithm like word2vec, in its skip-gram implementation specifically for this work, works. The idea is that flows, $$V_{f}$$ for vector representation, that occur within a window $$W_{f}$$, which can be modeled as "time" using timestamps from the capture. A visual representation of a single flow and window of flows can be seen below :

![](/img/flow_window_5.jpg)
*Windows of flows*

When we consider the conditional probabilities $$P(w\|f)$$ with a given set of flow captures **Captures** the goal is to set the parameters $$\theta$$ of $$P(w\|f;\theta)$$ so as to maximize the capture probability :

$$ \underset{\theta}{\operatorname{argmax}} \underset{f \in Captures}{\operatorname{\prod}} \left[\underset{w \in W_{f}}{\operatorname{\prod}} P(w \vert f;\theta)\right] $$

in this equation $$W_{f}$$ is a set of surrounding flows of flow $$f$$. Alternatively :

$$ \underset{\theta}{\operatorname{argmax}} \underset{(f, w) \in D}{\operatorname{\prod}} P(w \vert f;\theta) $$

Here $$D$$ is the set of all flow and window pairs we extract from the text.

The word2vec algorithm seems to capture an underlying phenomenon of written language that clusters words together according to their linguistic similarity, this can be seen in something like simple synonym analysis. The goal is to exploit this underlying "similarity" phenomenon with respect to co-occurence of flows in a given flow capture.

Each "time step", right now just being a subset of a given flow data set, is as a 'sentence' in the word2vec model. We should then be able to find flow "similarities" that exist within the context of flows. The idea is this "symilarity" will really just yield an occurence pattern over the flow data, much like word2vec does for written text.

Another part of the idea is much like in written text there are word / context, $$(w,c)$$, patterns that are discovered and exploited when running the algorithm over a given set of written language. There are common occurences and patterns that can be yielded from flow data, much like the occurences and patterns that are yielded from written text.

At the end of the embedding exercise we can use k-means to attempt to cluster flows, according to the embedding vectors that are produced through the word2vec algorithm. This should yield some sort of clustering of commonly occuring flows that have the same occurence measure in a given set of netflow captures. We can then use this data to measure against other, unseen, flows for future classification of "anamoly". I use that word loosely as this is strictly expirimental.

### Assumptions :

#### Maximizing the objective will result in good embeddings $$v_{f}  \forall w \in V$$

##### It is important to note with the above statment, with respect to time, is the assumption that the data I am operating from has already been ordered according to the tooling I used to acquire it_

## Skip-gram Negative Sampling

One of the other portions of the word2vec algorithm that I will be testing in this experiment will be negative sampling.

The objective of Skipgram with Negative Sampling is to maximize the the probability that $$(f,w)$$ came from the data $$D$$. This can be modeled as a distribution such that $$P(D=1\|f,w)$$ be the probability that $$(f,w)$$ came from the data and $$P(D=0\|f,w) = 1 - P(D=1\|f,w)$$ the probability that $$(f,w)$$ did not. 

The distribution is modeled as :

$$P(D=1|f,w) = \sigma(\vec{f} \cdot \vec{w}) = \frac{1}{1+e^{-\vec{f} \cdot \vec{w}}}$$

where $$\vec{f}$$ and $$\vec{w}$$, each a $$d$$-dimensional vector, are the model parameters to be learned.

The negative sampling tries to maximize $$P(D=1\|f,w)$$ for observed $$(f,w)$$ pairs while maximizing $$P(D=0\|f,w)$$ for stochastically sampled "negative" examples, under the assumption that selecting a context for a given word is likely to result in an unobserved $$(f,w)$$ pair.

SGNS's objective for a single $$(f,w)$$ output observation is then:

$$ E = \log \sigma(\vec{f} \cdot \vec{w}) + k \cdot \mathbb{E}_{w_{N} \sim P_{D}} [\log \sigma(\vec{-f} \cdot \vec{w}_N)] $$

where $$k$$ is the number of "negative" samples and $$w_{N}$$ is the sampled window, drawn according to the empirical unigram distribution $$P_{D}(w) = \frac{\text{#}w}{\|D\|}$$

Let's disassemble this objective function into its respective terms and put it back together again :

The term $$\log \sigma(\vec{f} \cdot \vec{w})$$, from above, is used to model the 

This object is then trained in an online fashion using stochastic gradient updated over the observed pairs in the corpus $$D$$. The goal objective then sums over the observed $$(f,w)$$ pairs in the corpus :

$$ \ell = \Sigma_{f \in V_{f}} \Sigma_{w \in V_{w}} \#(f,w)(\log \sigma(\vec{f} \cdot \vec{w}) + k \cdot \mathbb{E}_{w_{N} \sim P_{D}} [\log \sigma(\vec{-f} \cdot \vec{w}_N)]$$

Optimizing this objective groups flows that have similar embeddings, while scattering unobserved pairs.

##### TODO - further exploration : 

* Running true tuples of SRCIP, DSTIP, DSTPORT, and PROTO 
* Label included for now, need to figure out how to persist through pipeline without skewing results - need to figure out how to match up labeling to flow after word2vec has been run
* Implement timestamp window oriented 'sentence' creation, current implementation created same length flow 'sentences' for every $$f$$ flow


```python
# Method to slide window over dataframe of 
# flowdata and create "sentences"

def create_corpora(dataframe, window, corpus_count):
    corpus = []
    corpora = []
    begin = 0
    end = 0
    for i in range(corpus_count):
        while end <= window:
            end += 1
        else:
            corpus.append(dataframe[begin:(end-1)])
        begin = begin + window
        end = end + window
    corpora.append(corpus)
    return(corpora)
```


```python
corpora = create_corpora(subsample_cats, 30, 153333)
labels = create_corpora(subsample_labels, 30, 153333)
corpora_1 = create_corpora(subsample_cats_1, 30, 153333)
```


```python
# Convert all tuples created by previous create_corpora function
# to strings for use with tokenization which is then used in the
# word2vec algorithm below 

str_corpora = []

for corpus in corpora[0]:
    str_corpus = []
    for sentence in corpus.values.tolist():
        str_corpus.append(str(sentence).encode('utf-8'))
    str_corpora.append(str_corpus)
```




    "['94.44.127.113', '147.32.84.59', '6881', 'tcp']"




```python
# Here we train a model without using the negative sampling 
# hyperparameter. We will be using this for testing of 
# accuracy of model vs. using the negative sampling function

flow_model = gensim.models.Word2Vec(str_corpora, workers=23, 
                                    size=200, window=20, 
                                    min_count=1)
```


```python
# Here we train a model using the negative sampling which 
# we will then compare to the model above for the impact 
# that the negative sampling has on the clustering of flows

flow_model_sgns = gensim.models.Word2Vec(str_corpora, workers=23, 
                                         size=100, window=30, 
                                         negative=10, sample=5)
```

## Preliminary results - very rough, no real hyperparameter tunings / exploration, etc.

We can see below the results may prove to be useful with respect to certain labels present in the dataset, but not others. This may have to do with the raw occurence rates of certain flow and window #$$(f,w)$$ combinations vs. others. I use labels lightly as well as this will ultimately become an exercise of semi-supervised learning as it can sometimes be impossible for humans to interpret the results of an unsupervised learning task without any type of contextual insight, as labels can provide. In the case of written language, the "insight" that is provided is the fact that we know what the meanings of words are within the language and if they're clustering correctly, re: synonyms and antonyms, etc.

We can tune for this using subsampling above in the SGNS model. Which will we do next.

#### TODO:
* GridSearch for hyperparameters

Here we see that there is indeed a clustering that has happened with respect to the "From-Botnet-V42-UDP-DNS"


```python
# Test for flow similarity, preferrably a flow that has the botnet label

flow_model_1.most_similar("['147.32.84.165', '192.33.4.12', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']", topn=100)
```




    [("['147.32.84.165', '192.5.5.241', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9761667847633362),
     ("['147.32.84.165', '202.12.27.33', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9741541743278503),
     ("['147.32.84.165', '128.8.10.90', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.973616898059845),
     ("['147.32.84.165', '78.47.76.4', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9714504480361938),
     ("['147.32.84.165', '193.0.14.129', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9692395925521851),
     ("['147.32.84.165', '199.7.83.42', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9687032699584961),
     ("['147.32.84.165', '192.228.79.201', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9674479961395264),
     ("['147.32.84.165', '192.58.128.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9664252400398254),
     ("['147.32.84.165', '92.53.98.100', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9656703472137451),
     ("['147.32.84.165', '192.112.36.4', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9654155969619751),
     ("['147.32.84.165', '198.41.0.4', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9644977450370789),
     ("['147.32.84.165', '192.203.230.10', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9633801579475403),
     ("['147.32.84.165', '192.36.148.17', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9618400931358337),
     ("['147.32.84.165', '128.63.2.53', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.958657443523407),
     ("['147.32.84.165', '89.108.64.2', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9581812024116516),
     ("['147.32.84.165', '82.103.128.82', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9558319449424744),
     ("['147.32.84.165', '192.42.93.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9557339549064636),
     ("['147.32.84.165', '192.26.92.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9556182026863098),
     ("['147.32.84.165', '194.226.96.8', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9543852210044861),
     ("['147.32.84.165', '194.85.61.20', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.953228771686554),
     ("['147.32.84.165', '88.212.196.130', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9526883959770203),
     ("['147.32.84.165', '195.128.49.14', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9500119090080261),
     ("['147.32.84.165', '217.16.20.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9483109712600708),
     ("['147.32.84.165', '85.10.210.157', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9481122493743896),
     ("['147.32.84.165', '92.53.116.200', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9478355050086975),
     ("['147.32.84.165', '88.212.221.11', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9470769166946411),
     ("['147.32.84.165', '82.146.55.155', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9461535811424255),
     ("['147.32.84.165', '192.41.162.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9459192156791687),
     ("['147.32.84.165', '77.222.40.2', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9456772804260254),
     ("['147.32.84.165', '199.19.57.1', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.945094645023346),
     ("['147.32.84.165', '89.253.192.21', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9428556561470032),
     ("['147.32.84.165', '199.249.120.1', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9426734447479248),
     ("['147.32.84.165', '192.54.112.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9423930048942566),
     ("['147.32.84.165', '195.2.83.38', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9414822459220886),
     ("['147.32.84.165', '89.108.104.3', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9414548873901367),
     ("['147.32.84.165', '78.108.89.252', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9414442181587219),
     ("['147.32.84.165', '80.93.50.53', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9408544898033142),
     ("['147.32.84.165', '192.31.80.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9401237368583679),
     ("['147.32.84.165', '195.161.112.91', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.939973771572113),
     ("['147.32.84.165', '193.169.178.59', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9395020008087158),
     ("['147.32.84.165', '192.48.79.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9393561482429504),
     ("['147.32.84.165', '192.33.14.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9386749267578125),
     ("['147.32.84.165', '85.10.210.144', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9382632970809937),
     ("['147.32.84.165', '192.12.94.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9372074007987976),
     ("['147.32.84.165', '192.35.51.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9371063113212585),
     ("['147.32.84.165', '213.177.97.1', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9366581439971924),
     ("['147.32.84.165', '95.163.69.51', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9363342523574829),
     ("['147.32.84.165', '79.174.72.215', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.936087965965271),
     ("['147.32.84.165', '195.248.235.219', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9358514547348022),
     ("['147.32.84.165', '217.16.16.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9352473020553589),
     ("['147.32.84.165', '78.108.81.247', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9348022937774658),
     ("['147.32.84.165', '192.5.6.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.934520423412323),
     ("['147.32.84.165', '199.19.56.1', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.934291422367096),
     ("['147.32.84.165', '217.16.22.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9341065883636475),
     ("['147.32.84.165', '192.36.144.107', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9333975315093994),
     ("['147.32.84.165', '81.177.24.54', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9332102537155151),
     ("['147.32.84.165', '192.52.178.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9328247308731079),
     ("['147.32.84.165', '83.222.0.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9324507117271423),
     ("['147.32.84.165', '95.168.160.245', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9320420026779175),
     ("['147.32.84.165', '95.168.174.25', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9319052696228027),
     ("['147.32.84.165', '80.93.56.2', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9313104748725891),
     ("['147.32.84.165', '193.227.240.37', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9309282302856445),
     ("['147.32.84.165', '208.100.5.254', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9303311705589294),
     ("['147.32.84.165', '77.221.130.250', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9299085140228271),
     ("['147.32.84.165', '192.55.83.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9297108054161072),
     ("['147.32.84.165', '84.252.138.21', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9296650886535645),
     ("['147.32.84.165', '192.43.172.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.928945779800415),
     ("['147.32.84.165', '89.111.177.253', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9288318753242493),
     ("['147.32.84.165', '195.2.64.38', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9286403059959412),
     ("['147.32.84.165', '195.128.50.221', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9278726577758789),
     ("['147.32.84.165', '178.218.208.130', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9271195530891418),
     ("['147.32.84.165', '192.36.125.2', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9268661141395569),
     ("['147.32.84.165', '199.19.54.1', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9267032146453857),
     ("['147.32.84.165', '79.137.226.102', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9260225296020508),
     ("['147.32.84.165', '193.232.130.14', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9259271621704102),
     ("['147.32.84.165', '193.232.142.17', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9246711730957031),
     ("['147.32.84.165', '78.47.139.101', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.924452006816864),
     ("['147.32.84.165', '217.174.106.66', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9236003756523132),
     ("['147.32.84.165', '77.222.41.3', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9235631823539734),
     ("['147.32.84.165', '83.222.1.30', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9203209280967712),
     ("['147.32.84.165', '91.217.21.170', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9194321632385254),
     ("['147.32.84.165', '89.108.122.149', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.919041633605957),
     ("['147.32.84.165', '91.217.20.170', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9166457056999207),
     ("['147.32.84.165', '193.227.240.38', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9165226221084595),
     ("['147.32.84.165', '78.108.80.90', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9164752960205078),
     ("['147.32.84.165', '78.110.50.60', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.915980875492096),
     ("['147.32.84.165', '178.162.177.145', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9158682227134705),
     ("['147.32.84.165', '194.85.252.62', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.915434718132019),
     ("['147.32.84.165', '77.221.159.237', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9152796864509583),
     ("['147.32.84.165', '193.232.146.170', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9140732884407043),
     ("['147.32.84.165', '199.249.112.1', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9137414693832397),
     ("['147.32.84.165', '87.224.128.4', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9121522307395935),
     ("['147.32.84.165', '93.170.25.253', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9113649725914001),
     ("['147.32.84.165', '195.209.63.181', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9110352993011475),
     ("['147.32.84.165', '195.243.137.26', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9104222059249878),
     ("['147.32.84.165', '194.0.0.53', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9094029068946838),
     ("['147.32.84.165', '91.218.228.18', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9092046022415161),
     ("['147.32.84.165', '194.85.105.17', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9091553092002869),
     ("['147.32.84.165', '193.232.156.17', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9083877801895142),
     ("['147.32.84.165', '212.176.27.2', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      0.9074791669845581)]




```python
flow_model_1.most_similar("['147.32.84.165', '60.190.223.75', '888', 'tcp', 'flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data']")
```




    [("['217.66.146.105', '147.32.84.229', '443', 'tcp', 'flow=Background-TCP-Established']",
      0.970333993434906),
     ("['188.26.176.163', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      0.963600218296051),
     ("['114.75.11.242', '147.32.84.229', '80', 'tcp', 'flow=Background-TCP-Established']",
      0.9627201557159424),
     ("['147.32.86.96', '147.32.87.29', '0xb612', 'icmp', 'flow=Background']",
      0.9622609615325928),
     ("['195.234.241.9', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      0.9621870517730713),
     ("['41.130.66.62', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      0.9606925249099731),
     ("['131.104.149.212', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      0.9604771733283997),
     ("['147.32.84.59', '90.146.27.130', '46356', 'udp', 'flow=Background-Attempt-cmpgw-CVUT']",
      0.9597481489181519),
     ("['147.32.84.229', '78.141.179.11', '34046', 'udp', 'flow=Background-UDP-Established']",
      0.9597265720367432),
     ("['147.32.84.59', '114.40.199.143', '21323', 'udp', 'flow=Background-Established-cmpgw-CVUT']",
      0.9592392444610596)]



### Without label contained in the dataset

Here we run the same hyperparameters for the word2vec algorith, this time ignoring the label and not adding it to the "word" representations.


```python
flow_model_2 = gensim.models.Word2Vec(str_corpora, workers=23, size=100, window=30, negative=10, sample=5)
```


```python
flow_model_2.most_similar("['147.32.84.165', '192.33.4.12', '53', 'udp']")
```




    [("['147.32.84.165', '192.112.36.4', '53', 'udp']", 0.9759483337402344),
     ("['147.32.84.165', '193.0.14.129', '53', 'udp']", 0.9724588394165039),
     ("['147.32.84.165', '192.5.5.241', '53', 'udp']", 0.9721120595932007),
     ("['147.32.84.165', '128.8.10.90', '53', 'udp']", 0.9712154865264893),
     ("['147.32.84.165', '192.58.128.30', '53', 'udp']", 0.9697802662849426),
     ("['147.32.84.165', '192.36.148.17', '53', 'udp']", 0.9674890041351318),
     ("['147.32.84.165', '198.41.0.4', '53', 'udp']", 0.9672064185142517),
     ("['147.32.84.165', '199.7.83.42', '53', 'udp']", 0.9657577872276306),
     ("['147.32.84.165', '202.12.27.33', '53', 'udp']", 0.9610617160797119),
     ("['147.32.84.165', '192.203.230.10', '53', 'udp']", 0.9608649015426636)]




```python
flowdata[flowdata['DstAddr'].str.contains("192.112.36.4", na=False)].head(n=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StartTime</th>
      <th>Dur</th>
      <th>Proto</th>
      <th>SrcAddr</th>
      <th>Sport</th>
      <th>Dir</th>
      <th>DstAddr</th>
      <th>Dport</th>
      <th>State</th>
      <th>sTos</th>
      <th>dTos</th>
      <th>TotPkts</th>
      <th>TotBytes</th>
      <th>SrcBytes</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1264477</th>
      <td>2011/08/10 12:29:05.687373</td>
      <td>0.258197</td>
      <td>udp</td>
      <td>147.32.84.165</td>
      <td>2077</td>
      <td>&lt;-&gt;</td>
      <td>192.112.36.4</td>
      <td>53</td>
      <td>CON</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>528</td>
      <td>68</td>
      <td>flow=From-Botnet-V42-UDP-DNS</td>
    </tr>
    <tr>
      <th>1264673</th>
      <td>2011/08/10 12:29:06.093217</td>
      <td>0.258987</td>
      <td>udp</td>
      <td>147.32.84.165</td>
      <td>2077</td>
      <td>&lt;-&gt;</td>
      <td>192.112.36.4</td>
      <td>53</td>
      <td>CON</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>611</td>
      <td>77</td>
      <td>flow=From-Botnet-V42-UDP-DNS</td>
    </tr>
  </tbody>
</table>
</div>




```python
vocab_flow = []

for flow in flow_model_2.vocab.items():
    if re.search(r"192.112.36.4", flow[0]):
        vocab_flow.append(flow)
```


```python
vocab_flow
```




    [("['147.32.84.165', '192.112.36.4', '53', 'udp']",
      <gensim.models.word2vec.Vocab at 0x7f808752a350>)]




```python
flowdata[flowdata['DstAddr'].str.contains("192.5.5.241", na=False)].head(n=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StartTime</th>
      <th>Dur</th>
      <th>Proto</th>
      <th>SrcAddr</th>
      <th>Sport</th>
      <th>Dir</th>
      <th>DstAddr</th>
      <th>Dport</th>
      <th>State</th>
      <th>sTos</th>
      <th>dTos</th>
      <th>TotPkts</th>
      <th>TotBytes</th>
      <th>SrcBytes</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>401163</th>
      <td>2011/08/10 10:24:11.577652</td>
      <td>0.004420</td>
      <td>udp</td>
      <td>147.32.87.49</td>
      <td>65174</td>
      <td>&lt;-&gt;</td>
      <td>192.5.5.241</td>
      <td>53</td>
      <td>CON</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>636</td>
      <td>132</td>
      <td>flow=Background-UDP-Established</td>
    </tr>
    <tr>
      <th>1265468</th>
      <td>2011/08/10 12:29:12.214663</td>
      <td>0.002282</td>
      <td>udp</td>
      <td>147.32.84.165</td>
      <td>2077</td>
      <td>&lt;-&gt;</td>
      <td>192.5.5.241</td>
      <td>53</td>
      <td>CON</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>417</td>
      <td>70</td>
      <td>flow=From-Botnet-V42-UDP-DNS</td>
    </tr>
  </tbody>
</table>
</div>



### Aggregated flows, equivalent to "phrases"

The word2vec algorithm can also learn embeddings for phrases as well as single words for written language. The ideas I have surrounding "phrases" would be learning the embeddings for given windows of flows, if they were to present themselves in certain capacities within the captures flow data.

The current flow data that this notebook is based around are aggregated flows for bi-directional communication between endpoints. Exploiting something like capturing the 'phrase' of a flow, or thought another way, the bi-directional communication patterns that are contained within flow data might prove useful for application profiling, etc. through the use of application meta-data tracked through some sort of semi-supervised learning pipeline.


# Clustering

Now that we have some vector representations of occurences of flows within the captures that we have, we can run a clustering algorithm over them to see if we can humanly identify some of the groupings that have taken place. For this, we'll use kmeans within the scikit-learn package.

Kmeans has an objective function that intends to partition $$n$$ objects into $$k$$ clusters in which each object, $$n$$, belongs to the cluster with the nearest mean. This can be seen as :

$$ J = \sum_{j=1}^{k}\sum_{i=1}^{n} \| x_{i}^{(j)} - c_{j}\|^2 $$

Where $$\| x_{i}^{(j)} - c_{j}\|^2$$ is a chosen distance measure between a datapoint $$x^{j}_{i}$$ and the cluster center $$c{j}$$, is an indicator of the distance of the $$n$$ datapoints from their respective cluster $$k$$ centers. In this case, $$k$$ is a hyperparameter that can be used within the model to define how many cluster centroids should be trained over.

#### TODO :

* Limitation for arrays larger than 16GB due to an underlying dependency that numpy has, need to investigate - this is why I'm only running kmeans on a subset of the overall model learned above
* Dimensionality reduction of some kind over the data - 300 dimensional data isn't crazy high but might have some improved performance here as well.


```python
# Set k (number of clusters) to be 1/5 of the "vocabulary" size
# or an average of flows per cluster, this is a hyperparameter
# in kmeans that we can tweak later on

flow_vectors = flow_model_1.syn0[0:20000]
num_clusters = flow_vectors.shape[0] / 5

# Initialize k-means object and use it to extract centroids

kmeans_clustering = cluster.KMeans(n_clusters = num_clusters, init="k-means++", n_jobs=-1)
idx = kmeans_clustering.fit_predict(flow_vectors)

# Create a flow / Index dictionary, mapping "vocabulary words" to
# a cluster number

flow_centroid_map = dict(zip(flow_model_1.index2word, idx))
```


```python
#Find some botnet labels to use for exploration of data

import operator
sorted_clusters = sorted(flow_centroid_map.items(), key=operator.itemgetter(1))

botnets = []

for i in sorted_clusters:
    if re.search(r"Botnet", i[0]):
        botnets.append(i)
        
botnets[0:10]
```




    [("['147.32.84.165', '209.86.93.226', '25', 'tcp', 'flow=From-Botnet-V43-TCP-Attempt-SPAM']",
      3),
     ("['147.32.84.165', '192.33.4.12', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      14),
     ("['147.32.84.165', '85.214.220.206', '25', 'tcp', 'flow=From-Botnet-V42-TCP-Attempt-SPAM']",
      40),
     ("['147.32.84.165', '77.88.210.88', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      48),
     ("['147.32.84.165', '75.180.132.243', '25', 'tcp', 'flow=From-Botnet-V42-TCP-Attempt-SPAM']",
      49),
     ("['147.32.84.165', '67.23.231.68', '53', 'udp', 'flow=From-Botnet-V42-UDP-Attempt-DNS']",
      73),
     ("['147.32.84.165', '60.190.223.75', '888', 'tcp', 'flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data']",
      74),
     ("['147.32.84.165', '80.93.50.53', '53', 'udp', 'flow=From-Botnet-V42-UDP-DNS']",
      89),
     ("['147.32.84.165', '94.100.176.20', '25', 'tcp', 'flow=From-Botnet-V43-TCP-Attempt-SPAM']",
      91),
     ("['147.32.84.165', '74.125.159.27', '25', 'tcp', 'flow=From-Botnet-V42-TCP-Attempt-SPAM']",
      99)]




```python
# Look at members of clusters according to botnet memberships discovered above

cluster_members = []
for i in sorted_clusters:
    if i[1] == 73:
        cluster_members.append(i)
    
cluster_members[0:10]
```




    [("['147.32.84.59', '72.21.210.129', '80', 'tcp', 'flow=Background-Established-cmpgw-CVUT']",
      73),
     ("['62.162.92.225', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      73),
     ("['147.32.84.59', '208.88.186.10', '34021', 'udp', 'flow=Background-Established-cmpgw-CVUT']",
      73),
     ("['147.32.84.165', '67.23.231.68', '53', 'udp', 'flow=From-Botnet-V42-UDP-Attempt-DNS']",
      73),
     ("['200.148.213.27', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      73),
     ("['187.75.138.219', '147.32.84.229', '13363', 'udp', 'flow=Background-UDP-Established']",
      73)]



## Cluster visualization

Raw flow vectors $$V_{f}$$, created by word2vec, are embedded in dimensionality equivalent to the input layer of the shallow neural network that is used within the model. In our example we're using 

### t-SNE Visualization

Use t-SNE and matplotlib to visualize the clusters created using Word2Vec.

#### TODO :

* Brief explanation of the tSNE algorithm and how it handles compressing higher dimensional data into 2 or 3 dimension for visualization


```python
def perform_tsne(word_vector):
    tsne = manifold.TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(word_vector)
```


```python
#flow_model_reduced = TruncatedSVD(n_components=100, random_state=42).fit_transform(flow_model_1.syn0)
test_tsne = manifold.TSNE(n_components=2, learning_rate=50).fit_transform(flow_model_1.syn0[0:4000])
```


```python
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'), figsize=(10, 10))

x = test_tsne[:,0]
y = test_tsne[:,1]

mpld3_scatter = ax.scatter(x, y, cmap='Blues', c = y)
ax.grid(color='white', linestyle='solid')

labels = [v[0] for k,v in enumerate(flow_model_1.vocab.items()[0-4000:])]
tooltip = mpld3.plugins.PointLabelTooltip(mpld3_scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)
```


```python
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'), figsize=(10, 10))


mpld3_scatter = ax.scatter(tsne_objs[0][:, 0], tsne_objs[0][:, 1])
ax.grid(color='white', linestyle='solid')

#ax.set_title("Scatter Plot (with tooltips!)", size=20)

#labels = [v[0][0] for k,v in enumerate(sample)]
tooltip = mpld3.plugins.PointLabelTooltip(mpld3_scatter)
mpld3.plugins.connect(fig, tooltip)
```


```python
fig = plt.figure(figsize=(70, 70))
ax = plt.axes(frameon=False)
plt.setp(ax,xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(flow_model_embedded_1[:, 0], flow_model_embedded_1[:, 1], marker="x")

#for k,v in enumerate(flow_model.vocab.items()):
#    plt.annotate(v[0], flow_model_embedded_1[k])

plt.savefig('test2.eps', format='eps', dpi=600)
```

## Things left to research / validate / test


* Tune hyperparameters of models for all algorithms -- word2vec, kmeans, tSNE
* Find fixes for limitations of larger datasets for tooling that has dependencies on numpy -- kmeans, tSNE
