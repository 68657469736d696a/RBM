A simple Restricted Boltzmann Machine implementation in R

# Installation
The RBM package can easily be installed using the following command. Please make sure the devtools package is installed
```
devtools::install_github('sdverkoelen/RBM')
```

# Example implementation
```R

# Data availible at https://www.samverkoelen.com/data
data <- as.matrix(read.csv('mnist_sample.csv'));

#train RBM stack
model <- stack.rbm (hidden = c(150, 75, 30), data = train)

par(mfrow=c(1,2))
id <- 57

#Original
original <- matrix(data[ ,id], byrow = T, ncol = 16)
original <- t(original[16:1, 1:16]) #correct orientation
image(original, col = grey(seq(0, 1, 0.001)), main = 'Original')

#visible to hidden
features <- up.rbm(model, data)

#hidden back to visible
reconstructions <- down.rbm(model, features)

#Reconstruction
reconstruction <- matrix(reconstructions[id, ], byrow = T, ncol = 16)
reconstruction <- t(reconstruction[16:1, 1:16]) #correct orientation
image(reconstruction, col = grey(seq(0, 1, 0.001)), main = 'Reconstruction')
```
