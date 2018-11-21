# Author: Di YANG
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require(mxnet)
source("model.R")

lstnet.input <- function(data, batch.begin, batch.size)
{
  data.batch <- list()
  data.batch[["data"]] <- mx.nd.array(data$data[,,batch.begin:(batch.begin+batch.size-1)])
  data.batch[["label"]] <- mx.nd.array(data$label[,batch.begin:(batch.begin+batch.size-1)])
  return(data.batch)
}

is.param.name <- function(name) 
{ 
  return (grepl('weight$', name) || grepl('bias$', name) ||  
            grepl('gamma$', name) || grepl('beta$', name) ) 
} 

mx.model.init.params <- function(symbol, input.shape, initializer, ctx) 
{ 
  if (!is.mx.symbol(symbol)) 
    stop("symbol need to be MXSymbol") 
  
  slist <- symbol$infer.shape(input.shape) 
  
  if (is.null(slist)) 
    stop("Not enough information to get shapes") 
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE) 
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE) 
  return(list(arg.params=arg.params, aux.params=aux.params)) 
}

# Initialize the data iter
mx.model.init.iter.rnn <- function(X, y, batch.size, is.train) 
{
  if (is.MXDataIter(X)) return(X)
  shape <- dim(data)
  if (is.null(shape)) {
    num.data <- length(X)
  } else {
    ndim <- length(shape)
    num.data <- shape[[ndim]]
  }
  if (is.null(y)) {
    if (is.train) stop("Need to provide parameter y for training with R arrays.")
    y <- c(1:num.data) * 0
  }
  
  batch.size <- min(num.data, batch.size)
  
  return(mx.io.arrayiter(X, y, batch.size=batch.size, shuffle=is.train))
}

is.MXDataIter <- function(x) 
{
  inherits(x, "Rcpp_MXNativeDataIter") ||
    inherits(x, "Rcpp_MXArrayDataIter")
}

# check data and translate data into iterator if data is array/matrix
check.data <- function(data, batch.size, is.train) 
{
  if (!is.null(data) && !is.list(data) && !is.mx.dataiter(data)) 
  {
    stop("The dataset should be either a mx.io.DataIter or a R list")
  }
  if (is.list(data)) {
    if (is.null(data$data) || is.null(data$label)){
      stop("Please provide dataset as list(data=R.array, label=R.array)")
    }
    data <- mx.model.init.iter.rnn(data$data, data$label, batch.size=batch.size, is.train = is.train)
  }
  if (!is.null(data) && !data$iter.next()) {
    data$reset()
    if (!data$iter.next()) stop("Empty input")
  }
  return (data)
}

update.closure <- function(optimizer, weight, grad, state.list) 
{
  ulist <- lapply(seq_along(weight), function(i) {
    if (!is.null(grad[[i]])) {
      optimizer$update(i, weight[[i]], grad[[i]], state.list[[i]])
    } else {
      return(NULL)
    }
  })
  # update state list, use mutate assignment
  state.list <- lapply(ulist, function(x) {
    x$state
  })
  # return updated weight list
  weight.list <- lapply(ulist, function(x) {
    x$weight
  })
  return(weight.list)
}


GCN.trian.model <- function(model,
                            train.data,
                            valid.data,
                            num.epoch,
                            learning.rate,
                            P,
                            support,
                            adj,
                            random.neighbor,
                            wd,
                            clip_gradient = NULL,
                            optimizer = 'sgd',
                            lr_scheduler = NULL)
{
  m <- model
  batch.size <- m$batch.size
  input.size <- m$input.size
  num.layers <- m$num.layers
  
  opt <- mx.opt.create(optimizer, learning.rate = learning.rate,
                       wd = wd,
                       rescale.grad = (1/batch.size),
                       clip_gradient=clip_gradient,
                       lr_scheduler = lr_scheduler)
  
  cat('\014')
  for(epoch in 1:num.epoch){
    batch.counter <- 1
    cat(paste0('Training Epoch ', epoch, '\n'))
    
    for(batch.counter in 1:num.batch){
      batch.begin <- (batch.counter-1)*batch.size+1
      gcn.inputs <- Graph.receptive.fields.computation(batch.begin,
                                                       batch.size,
                                                       P,
                                                       support,
                                                       adj,
                                                       num.layers,
                                                       random.neighbor)
      
      
      
    }
    
  }
  
  
  
}









GCN.setup.model <- function(gcn.sym,
                            ctx = mx.ctx.default(),
                            K,
                            layer.vecs,
                            hidden.num,
                            input.size,
                            batch.size,
                            initializer=mx.init.uniform(0.01))
{
  arg.names <- gcn.sym$arguments
  input.shape <- list()
  support.shape1 <- 1
  
  for(name in arg.names){
    if(grepl('data$', name)){
      input.shape[[name]] <- c(input.size, layer.vecs[length(layer.vecs)])
    }
    else if( grepl('label$', name) )
    {
      input.shape[[name]] <- c(batch.size)
    }else{
      support.shape2 <- 1
      for(i in 1:K){
        variable1 <- paste0("support.",i,".gcn$")
        variable2 <- paste0("support.tilde.",i,".gcn")
        variable3 <- paste0("H.",i,"tilde")
        if(grepl(variable1, name)){
          input.shape[[name]] <- c(layer.vecs[i+1], layer.vecs[i])
        }
        if(grepl(variable2, name)){
          input.shape[[name]] <- c(layer.vecs[i+1], layer.vecs[i])
        }
        if(grepl(variable3, name)){
          input.shape[[name]] <- c(hidden.num[i],layer.vecs[i+1])
        }
      }
    }
  }
  
  params <- mx.model.init.params(symbol = gcn.sym, input.shape = input.shape, initializer = initializer, ctx = ctx)
  
  args <- input.shape
  args$symbol <- gcn.sym
  args$ctx <- ctx
  args$grad.req <- 'add'
  gcn.exec <- do.call(mx.simple.bind, args)
  
  mx.exec.update.arg.arrays(gcn.exec, params$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(gcn.exec, params$aux.params, match.name = TRUE)
  
  grad.arrays <- list()
  for (name in names(gcn.exec$ref.grad.arrays)) {
    if (is.param.name(name))
      grad.arrays[[name]] <- gcn.exec$ref.arg.arrays[[name]]*0
  }
  mx.exec.update.grad.arrays(gcn.exec, grad.arrays, match.name=TRUE)
  
  return (list(gcn.exec = gcn.exec, 
               symbol = gcn.sym,
               num.layers = K,
               layer.vecs = layer.vecs,
               batch.size = batch.size,
               input.size = input.size,
               layer.hidden = hidden.num))
  
}













