activation <- function(rbm_w, state) {
  1/(1+exp(-rbm_w %*% state))
}

sample_binomial <- function(mat) {
  dims=dim(mat)
  matrix(rbinom(prod(dims),size=1,prob=c(mat)),dims[1],dims[2])
}

#' Train a single layer RBM
#'
#' @param hidden number of units in the hidden layer
#' @param data matrix with training data
#' @param learning_rate a numeric value spacifying the learning rate 
#' @param epochs the number of iterations per batch
#' @param batch_size during one epoch a single batch will be trained. This number must be divisible by the number of training items
#' @param momentum  a numeric value spacifying the momentum 
#' @param verbose logical value for showing the current status in the console 
#'
#' @return RBM object
#'
#' @examples
#' # Data availible at https://www.samverkoelen.com/data
#' data <- as.matrix(read.csv('mnist_sample.csv'));
#' 
#' #train the single RBM
#' model = single.rbm(hidden = 30, data=data, learning_rate=.09, epochs=1000, batch_size=100, momentum=0.9)
#' 
#' par(mfrow=c(1,2))
#' id <- 57
#' 
#' #Original
#' original <- matrix(data[ ,id], byrow = T, ncol = 16)
#' original <- t(original[16:1, 1:16]) #correct orientation
#' image(original, col = grey(seq(0, 1, 0.001)), main = 'Original')
#' 
#' #visible to hidden
#' features <- up.rbm(model, data)
#' 
#' #hidden back to visible
#' reconstructions <- down.rbm(model, features)
#' 
#' #Reconstruction
#' reconstruction <- matrix(reconstructions[id, ], byrow = T, ncol = 16)
#' reconstruction <- t(reconstruction[16:1, 1:16]) #correct orientation
#' image(reconstruction, col = grey(seq(0, 1, 0.001)), main = 'Reconstruction')
#' @export
single.rbm <- function(hidden, data, learning_rate = .1, epochs = 1000, batch_size=100, momentum=0.9, verbose=T, custom_verbose = '') {
  n=dim(data)[2]
  p=dim(data)[1]
  
  #Just checking
  if (n %% batch_size != 0) { stop("batch size must be a multiple of the number of trainings data ") }
  if (hidden <= 0) { stop("At least 1 hidden node is required") }
  
  model = (matrix(runif(hidden*p),hidden,p) * 2 - 1) * 0.1
  momentum_speed = matrix(0,hidden,p)
  
  start = 1;
  for (epoch in 1:epochs) {
    if (verbose) {
        width <- options()$width / 2
        cat('\014Training RBM',custom_verbose,'\nvisible: ',p,' hidden: ',hidden,' learning rate: ',learning_rate,' momentum: ',momentum,' epochs: ',epochs, ' batch size: ',batch_size,'\n')
        cat('\n ',
            paste0(rep('=', round(epoch / epochs * width)), collapse = ''), 
            paste0(rep('.', width-round((epoch / epochs * width))), collapse = ''),
            ' ',
            paste0(round(epoch / epochs * 100), '% completed'), sep='')
      }
      
    batch           <- data[, start:(start + batch_size - 1)]
    start           <- (start + batch_size) %% n
    visible_data    <- sample_binomial(batch)
    H0              <- sample_binomial(activation(model, batch))
    vh0             <- H0 %*% t(batch)/dim(batch)[2]
    V1              <- sample_binomial(activation(t(model), H0))
    H1              <- activation(model, V1)
    vh1             <- H1 %*% t(V1)/dim(V1)[2]
    gradient        <- vh0-vh1
    momentum_speed  <- momentum * momentum_speed + gradient
    model           <- model + momentum_speed * learning_rate
  }

  single <- list(weights    = model, 
              learning_rate = learning_rate, 
              epochs        = epochs, 
              batch_size    = batch_size, 
              momentum      = momentum)
  class(single) <- 'rbm_single'
  return(single)
}

#' Train multiple layers of RBM's
#'
#' @param hidden a vector containing the number of hidden units per layer
#' @param data matrix with training data
#' @param learning_rate a numeric value spacifying the learning rate or a vector containing these values for each layer 
#' @param epochs the number of iterations per batch or a vector containing these values for each layer
#' @param batch_size during one epoch a single batch will be trained. This number must be divisible by the number of training items
#' @param momentum  a numeric value spacifying the momentum or a vector containing these values for each layer
#' @param verbose logical value for showing the current status in the console 
#'
#' @return RBM object
#'
#' @examples
#' # Data availible at https://www.samverkoelen.com/data
#' data <- as.matrix(read.csv('mnist_sample.csv'));
#' 
#' #train RBM stack
#' model <- multi.rbm (hidden = c(150, 75, 30), data = data)
#' 
#' par(mfrow=c(1,2))
#' id <- 57
#' 
#' #Original
#' original <- matrix(data[ ,id], byrow = T, ncol = 16)
#' original <- t(original[16:1, 1:16]) #correct orientation
#' image(original, col = grey(seq(0, 1, 0.001)), main = 'Original')
#' 
#' #visible to hidden
#' features <- up.rbm(model, data)
#' 
#' #hidden back to visible
#' reconstructions <- down.rbm(model, features)
#' 
#' #Reconstruction
#' reconstruction <- matrix(reconstructions[id, ], byrow = T, ncol = 16)
#' reconstruction <- t(reconstruction[16:1, 1:16]) #correct orientation
#' image(reconstruction, col = grey(seq(0, 1, 0.001)), main = 'Reconstruction')
#' @export
multi.rbm <- function(hidden, data, learning_rate = .1, epochs = 1000, batch_size=100, momentum=0.9, verbose=T){
  #Just checking
  if (length(hidden) <= 1) { stop("Please use single.rbm for training a single layer") }

  if(length(learning_rate) == 1){ learning_rate <- rep(learning_rate, length(hidden)) }
  if (length(hidden) != length(learning_rate)) { stop("The number of layers and learning rates does not match") }

  if(length(epochs) == 1){ epochs <- rep(epochs, length(hidden)) }
  if (length(hidden) != length(epochs)) { stop("The number of layers and epochs does not match") }
  
  if(length(momentum) == 1){ momentum <- rep(momentum, length(hidden)) }
  if (length(hidden) != length(momentum)) { stop("The number of layers and momentum does not match") }

  stack <- as.list(length(hidden))
  out   <- data

  for(i in 1:length(hidden)){
    stack[[i]] <- single.rbm(hidden[i], out, learning_rate[i], epochs[i], batch_size, momentum[i], verbose, paste('layer', i))
    out <- activation(stack[[i]]$weights, out)
  }
  class(stack) <- 'rbm_stack'
  return(stack)
}

#' Retrieve features from a dataset
#'
#' @param model an RBM object from the single.rbm or multi.rbm functions
#' @param data matrix with input data
#'
#' @return a matrix with fatures
#'
#' @examples
#' # Data availible at https://www.samverkoelen.com/data
#' data <- as.matrix(read.csv('mnist_sample.csv'));
#' 
#' #train RBM stack
#' model <- multi.rbm (hidden = c(150, 75, 30), data = data)
#' 
#' par(mfrow=c(1,2))
#' id <- 57
#' 
#' #Original
#' original <- matrix(data[ ,id], byrow = T, ncol = 16)
#' original <- t(original[16:1, 1:16]) #correct orientation
#' image(original, col = grey(seq(0, 1, 0.001)), main = 'Original')
#' 
#' #visible to hidden
#' features <- up.rbm(model, data)
#' 
#' #hidden back to visible
#' reconstructions <- down.rbm(model, features)
#' 
#' #Reconstruction
#' reconstruction <- matrix(reconstructions[id, ], byrow = T, ncol = 16)
#' reconstruction <- t(reconstruction[16:1, 1:16]) #correct orientation
#' image(reconstruction, col = grey(seq(0, 1, 0.001)), main = 'Reconstruction')
#' @export
up.rbm <- function(model, data){
  if(class(model) == 'rbm_single'){
    data <- activation(model$weights, data)
  }else if(class(model) == 'rbm_stack'){
    for(i in 1:length(model)){
      data <- activation(model[[i]]$weights, data)
    }
  }else{
    stop("Please use a valid model");
  }
  return(t(data))
}

#' Reconstruct original data using features
#'
#' @param model an RBM object from the single.rbm or multi.rbm functions
#' @param data matrix with features
#'
#' @return a matrix with reconstructions of the input features
#'
#' @examples
#' # Data availible at https://www.samverkoelen.com/data
#' data <- as.matrix(read.csv('mnist_sample.csv'));
#' 
#' #train RBM stack
#' model <- multi.rbm (hidden = c(150, 75, 30), data = data)
#' 
#' par(mfrow=c(1,2))
#' id <- 57
#' 
#' #Original
#' original <- matrix(data[ ,id], byrow = T, ncol = 16)
#' original <- t(original[16:1, 1:16]) #correct orientation
#' image(original, col = grey(seq(0, 1, 0.001)), main = 'Original')
#' 
#' #visible to hidden
#' features <- up.rbm(model, data)
#' 
#' #hidden back to visible
#' reconstructions <- down.rbm(model, features)
#' 
#' #Reconstruction
#' reconstruction <- matrix(reconstructions[id, ], byrow = T, ncol = 16)
#' reconstruction <- t(reconstruction[16:1, 1:16]) #correct orientation
#' image(reconstruction, col = grey(seq(0, 1, 0.001)), main = 'Reconstruction')
#' @export
down.rbm <- function(model, data){
  data <- t(data)
  if(class(model) == 'rbm_single'){
    data <- activation(t(model$weights), data)
  }else if(class(model) == 'rbm_stack'){
    for(i in length(model):1){
      data <- activation(t(model[[i]]$weights), data)
    }
  }else{
    stop("Please use a valid model");
  }
  return(t(data))
}