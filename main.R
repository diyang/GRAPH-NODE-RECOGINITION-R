require(mxnet)
setwd("~/Documents/graph_convolutional_network")
#source("I:/Desktop/R/graph_convolutional_network/model.R")
source("./utils.R")
source("./train.R")

graph.inputs <- loaddata.cora()

K <- 2
batch.begin <- 1
batch.size <- 1
P <- graph.inputs$P
support <- graph.inputs$Atilde
adj <- graph.inputs$adjmatrix
layer.tP <- NULL
random.neighbor <- c(10,5)
layer.vecs <- c(100, 400, 1600)
input.size <- dim(graph.inputs$features)[2]

gcn.inputs <- Graph.receptive.fields.computation(batch.begin,
                                                 batch.size,
                                                 P,
                                                 support,
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

