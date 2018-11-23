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

require('mxnet')

Graph.Convolution <- function(data,
                              neighbors,
                              tP, 
                              num.hidden,
                              dropout = 0)
{
  if(dropout > 0){
    data <- mx.symbol.Dropout(data=data, p=dropout)
  }
  data.aggerator <- mx.symbol.dot(tP, neighbors)
  conv.input <- mx.symbol.Concat(data = c(data, data.aggerator), num.args = 2, dim = 1)
  graph.output <- mx.symbol.FullyConnected(data=conv.input, num_hidden = num.hidden)
  graph.activation <- mx.symbol.Activation(data=graph.output, act.type='relu')
  graph.L2norm <- mx.symbol.L2Normalization(graph.activation)
  return(graph.L2norm)
}

GCN.two.layer.model <- function(num.hidden, num.label, dropout = 0){
  label <- mx.symbol.Variable('label')
  layer.tP <- list()
  layer.H  <- list()
  K <- length(num.hidden)
  for(i in 1:K){
    layer.tP[[i]] <- mx.symbol.Variable(paste0("P.",i,".tilde"))
    layer.H[[i]]  <- mx.symbol.Variable(paste0("H.",i,".tilde"))
  }
  layer.H[[K+1]]  <- mx.symbol.Variable(paste0("H.",(K+1),".tilde"))
  
  layer.outputs <- list()
  for(i in K:1){
    gcn.input <- layer.H[[i]]
    if(i == K){
      neighbour.input <- layer.H[[i+1]]
    }else{
      neighbour.input <- layer.outputs[[i+1]]
    }
    layer.outputs[[i]] <- Graph.Convolution(data=gcn.input,
                                            neighbors = neighbour.input,
                                            tP=layer.tP[[i]], 
                                            num.hidden = num.hidden[i])
  }
  
  fc <- mx.symbol.FullyConnected(data=layer.outputs[[1]], num.hidden=num.label)
  loss.all <- mx.symbol.SoftmaxOutput(data=fc, label=label, name="sm")
  return(loss.all)
}