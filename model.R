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
                              data.bar,
                              tP, 
                              P, 
                              num.hidden,
                              dropout = 0)
{
  delta.data <- data-data.bar
  part1 <- mx.symbol.dot(tP, delta.data)
  part2 <- mx.symbol.dot(P, data)
  conv.input <- (part1 + part2)
  graph.output <- mx.symbol.FullyConnected(data=conv.input, num_hidden = num.hidden)
  output <- mx.symbol.Activation(data=graph.output, act.type='relu')
  return(output)
}

GCN.two.layer.model <- function(num.hidden1, num.hidden2){
  data  <- mx.symbol.Variable('data')
  label <- mx.symbol.Variable('label')
  layer.P  <- list()
  layer.tP <- list()
  layer.H  <- list()
  for(i in 1:2){
    layer.P[[i]]  <- mx.symbol.Variable(paste0("support.",i,".gcn"))
    layer.tP[[i]] <- mx.symbol.Variable(paste0("support.tilde.",i,".gcn"))
    layer.H[[i]]  <- mx.symbol.Variable(paste0("H.",i,"tilde"))
  }
  layer1.out <- Graph.Convolution(data=data,
                                  data.bar = layer.H[[2]],
                                  tP=layer.tP[[2]], 
                                  P=layer.P[[2]], 
                                  num.hidden = num.hidden1)
  layer2.out <- Graph.Convolution(data=layer1.out,
                                  data.bar = layer.H[[1]],
                                  tP=layer.tP[[1]], 
                                  P=layer.P[[1]], 
                                  num.hidden = num.hidden2)
  GCN.model <- mx.symbol.SoftmaxOutput(data=layer2.out, label=label)
  return(GCN.model)
}