# Adaptive Cut

This repository contains code for the "Adaptive Cut" method, which uses an adaptive MCMC algorithm to optimize multilevel cuts of a dendrogram. The algorithm is implemented in Python.

## Installation

To use the Adaptive Cut method, you need to have Python 3.x installed. You also need to install the following dependencies:



You can install these dependencies using pip:


## Usage

To use the Adaptive Cut method, you need to provide a dendrogram as input. The dendrogram can be represented as a linkage matrix, which can be generated using the `scipy.cluster.hierarchy.linkage()` function. Here's an example code snippet to generate a dendrogram from a distance matrix:

```python
from scipy.cluster.hierarchy import linkage

# generate a distance matrix
dist_matrix = ...

# generate a dendrogram from the distance matrix
linkage_matrix = linkage(dist_matrix, method='ward')
```

## Examples

## References
