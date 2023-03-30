Adaptive Cut

This repository contains code for the "Adaptive Cut" method, which uses an adaptive MCMC algorithm to optimize multilevel cuts of a dendrogram. The algorithm is implemented in Python.
Installation

To use the Adaptive Cut method, you need to have Python 3.x installed. You also need to install the following dependencies:

    NumPy
    SciPy
    NetworkX
    Matplotlib

You can install these dependencies using pip:

pip install numpy scipy networkx matplotlib

Usage

To use the Adaptive Cut method, you need to provide a dendrogram as input. The dendrogram can be represented as a linkage matrix, which can be generated using the scipy.cluster.hierarchy.linkage() function. Here's an example code snippet to generate a dendrogram from a distance matrix:

python

from scipy.cluster.hierarchy import linkage

# generate a distance matrix
dist_matrix = ...

# generate a dendrogram from the distance matrix
linkage_matrix = linkage(dist_matrix, method='ward')

Once you have a dendrogram, you can use the Adaptive Cut method to perform a multilevel cut. Here's an example code snippet to perform a cut using the Adaptive Cut method:

python

from adaptive_cut import adaptive_cut

# perform a multilevel cut using the Adaptive Cut method
labels = adaptive_cut(linkage_matrix, n_levels=3, n_iterations=1000)

In this example, we use the adaptive_cut() function from the adaptive_cut module to perform a multilevel cut of the dendrogram represented by the linkage_matrix. We specify that we want to perform a 3-level cut and run the adaptive MCMC algorithm for 1000 iterations. The adaptive_cut() function returns a list of community labels for each node in the dendrogram.
Examples

The examples directory contains example Python scripts that demonstrate how to use the Adaptive Cut method on various datasets. To run an example, simply run the corresponding Python script:

bash

python examples/example1.py

License

This code is released under the MIT License. See LICENSE for more information.
References

    Some reference papers go here
