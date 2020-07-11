# python-Combining-Content-and-Link-for-Classification-using-Matrix-Factorization
self-implementation of "Combining Content and Link for Classification using Matrix Factorization". both link-content-MF &amp; link-content-sup-MF version
# What it it?<br>
It is my implementation of "Combining Content and Link for Classification using Matrix Factorization"<br>
link-content-MF version in link_content_MF.py<br>
link-content-sup-MF version in link_content_sup_MF.py<br>
I use utils.py from [pyGAT](https://github.com/Diego999/pyGAT) to process Cora dataset<br>
(extract adjacency matrix, content feature matrix, labels, indexes for training, testing and infering)

# What it can do?<br>
It helps extract features in "Latent Factor Space" from raw content matrix and link matrix<br>
See more in the [paper](https://dl.acm.org/doi/pdf/10.1145/1277741.1277825)<br>

# How to use it?<br>
If you're using Cora dataset, just run it.<br>
Or else you need to change the path and name for your own dataset.<br> 

# Problem to solve
I haven't find some proper hyper parameters to get a good fit<br>
Let me explain it in detail:I ran my code on Cora dataset and tried a dozen of different parameters on,
for example, alpha, learning rate, lamda, and expecially the initializations of U, V, Z and W which have a significant impact on performance.
And there are two kind of bad results:<br>
1. gradients explodes<br>
2. loss converges in a relatively high value<br>
the reasons are still not clear to me......
