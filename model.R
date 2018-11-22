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
                              data.neighbor,
                              tP, 
                              num.hidden,
                              dropout = 0)
{
  data.aggerator <- mx.symbol.dot(tP, data.neighbor)
  conv.input <- mx.symbol.concat(data = c(data, data.aggerator), num.args = 2, dim = 1)
  graph.output <- mx.symbol.FullyConnected(data=conv.input, num_hidden = num.hidden)
  output <- mx.symbol.Activation(data=graph.output, act.type='relu')
  return(output)
}

GCN.two.layer.model <- function(num.hidden){
  data  <- mx.symbol.Variable('data')
  label <- mx.symbol.Variable('label')
  layer.tP <- list()
  layer.H  <- list()
  K <- length(num.hidden)
  for(i in 1:K){
    layer.tP[[i]] <- mx.symbol.Variable(paste0("support.tilde.",i,".gcn"))
    layer.H[[i]]  <- mx.symbol.Variable(paste0("H.",i,"tilde"))
  }
  
  layer.outputs <- list()
  for(i in 1:K){
    idx <- K-i+1
    if(i == 1){
      gcn.input <- data
    }else{
      gcn.input <- layer.outputs[[i-1]]
    }
    layer.outputs[[i]] <- Graph.Convolution(data=gcn.input,
                                            data.bar = layer.H[[idx]],
                                            tP=layer.tP[[idx]], 
                                            num.hidden = num.hidden[idx])
  }
  GCN.model <- mx.symbol.SoftmaxOutput(data=layer.outputs[[K]], label=label)
  return(GCN.model)
}