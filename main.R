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
                             random.neighbor,
                             hidden.num = c(25,10),
                             input.size,
                             batch.size,
                             mx.ctx.default(),
                             mx.init.uniform(0.01))

data1 <- mx.symbol.Variable('data1')
data2 <- mx.symbol.Variable('data2')
conv.input <- mx.symbol.concat(data = c(data1, data2), num.args = 2, dim = 1)

input.shape <- list()
input.shape[['data1']] <- c(50,30)
input.shape[['data2']] <- c(20,30)

ctx <- mx.ctx.default()
initializer<-mx.init.uniform(0.01)
params <- mx.model.init.params(symbol = conv.input, input.shape = input.shape, initializer = initializer, ctx = ctx)


