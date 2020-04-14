"""
This module focuses on the final step in the proposed word recognition model.
That is estimating a word, based on some lexicon, estimated PHOC or neural codes and a process of measure.


The process of measure may vary, but the estimator is always provided with a lexicon of words.
We define the construct of a generic estimator in the **base** module.

|

The **cosine** and **euclidean** modules build on a metric driven nearest neighbour approach,
while the **prob** interprets our estimated PHOC/ neural codes as probabilities and utilizes the PRM-Scores (see: A Probabilistic Retrieval Model for Word Spotting Based on Direct Attribute Prediction).
The **cca** module is unique in the sense that the process of measure is defined by transforming PHOC and estimated PHOC/ neural codes into a subspace, using regularized CCA, before applying a nearest neighbour search.
"""
