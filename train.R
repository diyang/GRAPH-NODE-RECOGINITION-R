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

calc.nll <- function(seq.label.probs, batch.size) {
  seq.label.probs <- na.omit(seq.label.probs)
  nll =  sum(seq.label.probs) / batch.size
  return (nll)
}

GCN.trian.model <- function(model,
                            graph.input,
                            nodes.train.pool,
                            nodes.valid.pool = NULL,
                            num.epoch,
                            learning.rate,
                            weight.decay,
                            clip.gradient = NULL,
                            optimizer = 'sgd',
                            lr.scheduler = NULL)
{
  m <- model
  batch.size <- m$batch.size
  input.size <- m$input.size
  layer.vecs <- m$layer.vecs
  random.neighbor <- m$random.neighbor
  K <- m$K
  
  opt <- mx.opt.create(optimizer, learning.rate = learning.rate,
                       wd = weight.decay,
                       rescale.grad = (1/batch.size),
                       clip_gradient=clip.gradient,
                       lr_scheduler = lr.scheduler)
  
  state.list <- lapply(seq_along(m$gcn.exec$ref.arg.arrays), function(i) {
    if (is.null(m$gcn.exec$ref.arg.arrays[[i]])) return(NULL)
    opt$create.state(i, m$gcn.exec$ref.arg.arrays[[i]])
  })

  cat('\014')
  for(epoch in 1:num.epoch){
    cat(paste0('Training Epoch ', epoch, '\n'))
    train.nll <- 0
    ##################
    # batch training #
    ##################
    for(batch.counter in 1:num.batch){
      # gcn input data preparation
      batch.begin <- (batch.counter-1)*batch.size+1
      nodes.train.batch <- nodes.train.pool[batch.begin:(batch.begin+batch.size-1)] 
      gcn.layer.input <- Graph.receptive.fields.computation(nodes.train.batch, graph.input$P, graph.input$adjmatrix, random.neighbor)

      gcn.train.data <- list()
      for(i in 1:K){
        variable.P <- paste0("P.",i,".tilde")
        variable.H <- paste0("H.",i,".tilde")
        
        gcn.train.data[[variable.P]] <- mx.nd.array(t(gcn.layer.input$tP[[i]]))
        
        if(length(gcn.layer.input$H[[i]]) == layer.vecs[i]){
          gcn.train.data[[variable.H]] <- mx.nd.array(t(graph.input$features$data[gcn.layer.input$H[[i]],]))
        }else{
          #padding layer inputs
          offset.vecs <- layer.vecs[i] - length(gcn.layer.input$H[[i]])
          padding <- matrix(0, offset.vecs, input.size)
          gcn.train.data[[variable.H]] <- mx.nd.array(t(rbind(as.matrix(graph.input$features$data[gcn.layer.input$H[[i]],]),padding)))
        }
      }
      
      if(length(gcn.layer.input$H[[K+1]]) == layer.vecs[K+1]){
        gcn.train.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(graph.input$features$data[gcn.layer.input$H[[(K+1)]],]))
      }else{
        #padding layer inputs
        offset.vecs <- layer.vecs[K+1] - length(gcn.layer.input$H[[K+1]])
        padding <- matrix(0, offset.vecs, input.size)
        gcn.train.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(rbind(as.matrix(graph.input$features$data[gcn.layer.input$H[[(K+1)]],]),padding)))
      }
      gcn.train.data[["label"]] <- mx.nd.array(graph.input$features$label[gcn.layer.input$H[[1]]])
      
      
      mx.exec.update.arg.arrays(m$gcn.exec, gcn.train.data, match.name = TRUE)
      mx.exec.forward(m$gcn.exec, is.train = TRUE)
      mx.exec.backward(m$gcn.exec)
      arg.blocks <- update.closure(optimizer = opt, weight = m$gcn.exec$ref.arg.arrays, 
                                   grad = m$gcn.exec$ref.grad.arrays, state.list = state.list)
      mx.exec.update.arg.arrays(m$gcn.exec, arg.blocks, skip.null=TRUE)
      
      label.probs <- mx.nd.choose.element.0index(m$gcn.exec$ref.outputs[["sm_output"]], m$gcn.exec$ref.arg.arrays[["label"]])
      train.nll <- train.nll + calc.nll(as.array(label.probs), batch.size)
      cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] Trian: NLL=", train.nll / batch.counter,"\n"))
    }
    ####################
    # batch validating #
    ####################
    if(!is.null(nodes.valid.pool)){
      cat("\n")
      cat("Validating \n")
      for(batch.counter in 1:num.batch.valid){
        # gcn input data preparation
        batch.begin <- (batch.counter-1)*batch.size+1
        nodes.valid.batch <- nodes.valid.pool[batch.begin:(batch.begin+batch.size-1)] 
        gcn.inputs <- Graph.receptive.fields.computation(nodes.valid.batch, graph.input$P, graph.input$adjmatrix, random.neighbor)
        gcn.valid.data <- list()
        for(i in 1:K){
          variable.P <- paste0("P.",i,".tilde")
          variable.H <- paste0("H.",i,".tilde")
          
          gcn.valid.data[[variable.P]] <- mx.nd.array(t(gcn.inputs$tP[[i]]))
          
          if(length(gcn.inputs$H[[i]]) == layer.vecs[i]){
            gcn.valid.data[[variable.H]] <- mx.nd.array(t(train.data[gcn.inputs$H[[i]],]))
          }else{
            #padding layer inputs
            offset.vecs <- layer.vecs[i] - length(gcn.inputs$H[[i]])
            padding <- matrix(0, offset.vecs, input.size)
            gcn.valid.data[[variable.H]] <- mx.nd.array(t(rbind(as.matrix(train.data[gcn.inputs$H[[i]],]),padding)))
          }
        }
        
        if(length(gcn.inputs$H[[K+1]]) == layer.vecs[K+1]){
          gcn.valid.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(train.data[gcn.inputs$H[[(K+1)]],]))
        }else{
          #padding layer inputs
          offset.vecs <- layer.vecs[K+1] - length(gcn.inputs$H[[K+1]])
          padding <- matrix(0, offset.vecs, input.size)
          gcn.valid.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(rbind(as.matrix(train.data[gcn.inputs$H[[(K+1)]],]),padding)))
        }
        
        mx.exec.update.arg.arrays(m$gcn.exec, gcn.valid.data, match.name = TRUE)
        mx.exec.forward(m$gcn.exec, is.train = FALSE)
        
        label.probs <- mx.nd.choose.element.0index(m$gcn.exec$ref.outputs[["sm_output"]], m$gcn.exec$ref.arg.arrays[["label"]])
        valid.nll <- valid.nll + calc.nll(as.array(label.probs), batch.size)
        cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] Valid: NLL=", valid.nll / batch.counter,"\n"))
      }
    }
    cat("\n")
  }
  return(m)
}

GCN.setup.model <- function(gcn.sym,
                            random.neighbor,
                            input.size,
                            batch.size,
                            ctx = mx.ctx.default(),
                            initializer=mx.init.uniform(0.01))
{
  arg.names <- gcn.sym$arguments
  input.shape <- list()
  support.shape1 <- 1
  K <- length(random.neighbor)
  
  layer.vecs <- c(batch.size)
  for(i in 1:K){
    layer.vecs[i+1] <- layer.vecs[i]*random.neighbor[i]
  }
  
  for(name in arg.names){
    if( grepl('label$', name) )
    {
      input.shape[[name]] <- c(batch.size)
    }else{
      for(i in K:1){
        variable.P <- paste0("P.",i,".tilde")
        variable.H <- paste0("H.",(i+1),".tilde")
        if(grepl(variable.P, name)){
          input.shape[[name]] <- c(layer.vecs[i+1], layer.vecs[i])
        }
        if(grepl(variable.H, name)){
          input.shape[[name]] <- c(input.size,layer.vecs[i+1])
        }
      }
      variable.H <- paste0("H.",1,".tilde")
      if(grepl(variable.H, name)){
        input.shape[[name]] <- c(input.size,layer.vecs[1])
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
               K = K,
               random.neighbor = random.neighbor,
               layer.vecs = layer.vecs,
               batch.size = batch.size,
               input.size = input.size))
  
}













