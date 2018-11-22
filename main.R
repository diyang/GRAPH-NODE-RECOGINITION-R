require(mxnet)
#windows
#setwd("I:/Desktop/R/SAGE-GRAPH-R")
#source("I:/Desktop/R/SAGE-GRAPH-R/model.R")
#source("I:/Desktop/R/SAGE-GRAPH-R/utils.R")
#source("I:/Desktop/R/SAGE-GRAPH-R/train.R")

#Mac
setwd("~/Documents/SAGE-GRAPH-R")
source("./model.R")
source("./utils.R")
source("./train.R")

graph.inputs <- loaddata.cora()

K <- 2
batch.begin <- 1
batch.size <- 1
P <- graph.inputs$P
adj <- graph.inputs$adjmatrix
layer.tP <- NULL
random.neighbor <- c(10,5)
layer.vecs <- c(100, 400, 1600)
input.size <- dim(graph.inputs$features)[2]

gcn.inputs <- Graph.receptive.fields.computation(batch.begin,
                                                 batch.size,
                                                 P,
                                                 adj,
                                                 K,
                                                 random.neighbor)

gcn.sym <- GCN.two.layer.model(num.hidden1 = 25, 
                               num.hidden2 = 10)

gcn.model <- GCN.setup.model(gcn.sym,
                             mx.ctx.default(),
                             2,
                             layer.vecs,
                             c(25,50),
                             input.size,
                             batch.size,
                             mx.init.uniform(0.01))