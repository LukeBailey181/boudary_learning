
* Get 10 vaes,  1 for each digit of mnist (can look online pretrained ones
or a pretrained a single one). 

Look for an importance weighted autoencoder 

* Use them to get boundaries of each class of mnist.
* Train classifier on these boundaries, just like I did on 
previous experiment 

* Start with 2-digits of mnist before building up - 3 and 8 for 
example because they are difficult.

* Plot digits in lower dimensional space - can use pcs or can plot single dimensions 
    * If these look vaguely gaussian we can calculate the mean and gaussian empirically 
    * Then we can refer to boundary points as least likely from this distribution 
    * We can also do this as a nearest neighbors from the centroid 
    * Third way to pick boundary points is the ELBO - do this by class (incase some 
    classes have much lower elbo values)
    * Side idea of training all 10 VAEs

