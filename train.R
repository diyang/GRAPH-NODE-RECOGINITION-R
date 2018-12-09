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
#source("Optimizer.R")

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
                            learning.rate = 0.01,
                            weight.decay = 0,
                            clip.gradient = 1,
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
  
  opt.updater <- mx.opt.get.updater(opt, m$gcn.exec$ref.arg.arrays)

  cat('\014')
  for(epoch in 1:num.epoch){
    cat(paste0('Training Epoch ', epoch, '\n'))
    
    ##################
    # batch training #
    ##################
    train.nll <- 0
    num.batch.train <- floor(length(nodes.train.pool)/batch.size)
    train.metric <- mx.metric.accuracy$init()
    for(batch.counter in 1:num.batch.train){
      # gcn input data preparation
      batch.begin <- (batch.counter-1)*batch.size+1
      nodes.train.batch <- nodes.train.pool[batch.begin:(batch.begin+batch.size-1)] 
      gcn.train.input <- Graph.receptive.fields.computation(nodes.train.batch, graph.input$P, graph.input$adjmatrix, random.neighbor)

      gcn.train.data <- list()
      for(i in 1:K){
        variable.P <- paste0("P.",i,".tilde")
        variable.H <- paste0("H.",i,".tilde")
        
        gcn.train.data[[variable.P]] <- mx.nd.array(t(gcn.train.input$tP[[i]]))
        
        if(length(gcn.train.input$H[[i]]) == layer.vecs[i]){
          gcn.train.data[[variable.H]] <- mx.nd.array(t(graph.input$features$data[gcn.train.input$H[[i]],]))
        }else{
          #padding layer inputs
          offset.vecs <- layer.vecs[i] - length(gcn.train.input$H[[i]])
          padding <- matrix(0, offset.vecs, input.size)
          gcn.train.data[[variable.H]] <- mx.nd.array(t(rbind(as.matrix(graph.input$features$data[gcn.train.input$H[[i]],]),padding)))
        }
      }
      
      if(length(gcn.train.input$H[[K+1]]) == layer.vecs[K+1]){
        gcn.train.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(graph.input$features$data[gcn.train.input$H[[(K+1)]],]))
      }else{
        #padding layer inputs
        offset.vecs <- layer.vecs[K+1] - length(gcn.train.input$H[[K+1]])
        padding <- matrix(0, offset.vecs, input.size)
        gcn.train.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(rbind(as.matrix(graph.input$features$data[gcn.train.input$H[[(K+1)]],]),padding)))
      }
      gcn.train.data[["label"]] <- mx.nd.array(graph.input$features$label[gcn.train.input$H[[1]]])
      
      mx.exec.update.arg.arrays(m$gcn.exec, gcn.train.data, match.name = TRUE)
      mx.exec.forward(m$gcn.exec, is.train = TRUE)
      mx.exec.backward(m$gcn.exec)
      arg.blocks <- opt.updater(weight = m$gcn.exec$ref.arg.arrays, grad = m$gcn.exec$ref.grad.arrays)
      mx.exec.update.arg.arrays(m$gcn.exec, arg.blocks, skip.null=TRUE)

      train.metric <- mx.metric.accuracy$update(m$gcn.exec$ref.arg.arrays[["label"]], m$gcn.exec$ref.outputs[["sm_output"]], train.metric)
      result <- mx.metric.accuracy$get(train.metric)
      #cat(paste0("[", epoch, "] Train-", result$name, "=", result$value, "\n"))
      #label.probs <- mx.nd.choose.element.0index(m$gcn.exec$ref.outputs[["sm_output"]], m$gcn.exec$ref.arg.arrays[["label"]])
      #train.nll <- train.nll + calc.nll(as.array(label.probs), batch.size)
      cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] Trian: ",result$name,"=", result$value,"\n"))
    }
    result <- mx.metric.accuracy$get(train.metric)
    cat(paste0("[", epoch, "] Train-", result$name, "=", result$value, "\n"))
    
    ####################
    # batch validating #
    ####################
    if(!is.null(nodes.valid.pool)){
      cat("\n")
      cat("Validating \n")
      valid.nll <- 0
      num.batch.valid <- floor(length(nodes.valid.pool)/batch.size)
      eval.metric <- mx.metric.accuracy$init()
      for(batch.counter in 1:num.batch.valid){
        # gcn input data preparation
        batch.begin <- (batch.counter-1)*batch.size+1
        nodes.valid.batch <- nodes.valid.pool[batch.begin:(batch.begin+batch.size-1)] 
        gcn.valid.input <- Graph.receptive.fields.computation(nodes.valid.batch, graph.input$P, graph.input$adjmatrix, random.neighbor)
        
        gcn.valid.data <- list()
        for(i in 1:K){
          variable.P <- paste0("P.",i,".tilde")
          variable.H <- paste0("H.",i,".tilde")
          
          gcn.valid.data[[variable.P]] <- mx.nd.array(t(gcn.valid.input$tP[[i]]))
          
          if(length(gcn.valid.input$H[[i]]) == layer.vecs[i]){
            gcn.valid.data[[variable.H]] <- mx.nd.array(t(graph.input$features$data[gcn.valid.input$H[[i]],]))
          }else{
            #padding layer inputs
            offset.vecs <- layer.vecs[i] - length(gcn.valid.input$H[[i]])
            padding <- matrix(0, offset.vecs, input.size)
            gcn.valid.data[[variable.H]] <- mx.nd.array(t(rbind(as.matrix(graph.input$features$data[gcn.valid.input$H[[i]],]),padding)))
          }
        }
        
        if(length(gcn.valid.input$H[[K+1]]) == layer.vecs[K+1]){
          gcn.valid.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(graph.input$features$data[gcn.valid.input$H[[(K+1)]],]))
        }else{
          #padding layer inputs
          offset.vecs <- layer.vecs[K+1] - length(gcn.valid.input$H[[K+1]])
          padding <- matrix(0, offset.vecs, input.size)
          gcn.valid.data[[paste0("H.",(K+1),".tilde")]] <- mx.nd.array(t(rbind(as.matrix(graph.input$features$data[gcn.valid.input$H[[(K+1)]],]),padding)))
        }
        gcn.valid.data[["label"]] <- mx.nd.array(graph.input$features$label[gcn.valid.input$H[[1]]])
      
        mx.exec.update.arg.arrays(m$gcn.exec, gcn.valid.data, match.name = TRUE)
        mx.exec.forward(m$gcn.exec, is.train = FALSE)
        
        eval.metric <- mx.metric.accuracy$update(m$gcn.exec$ref.arg.arrays[["label"]], m$gcn.exec$ref.outputs[["sm_output"]], eval.metric)
        result <- mx.metric.accuracy$get(eval.metric)
        #label.probs <- mx.nd.choose.element.0index(m$gcn.exec$ref.outputs[["sm_output"]], m$gcn.exec$ref.arg.arrays[["label"]])
        #valid.nll <- valid.nll + calc.nll(as.array(label.probs), batch.size)
        cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] Valid: ",result$name,"=", result$value,"\n"))
      }
      result <- mx.metric.accuracy$get(eval.metric)
      cat(paste0("[", epoch, "] Valid-", result$name, "=", result$value, "\n"))
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
  args$grad.req <- 'write'
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