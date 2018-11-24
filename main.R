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

graph.input <- loaddata.cora()
level.label <- unique(graph.input$content$class)
num.label <- length(unique(level.label))
label.text <-graph.input$content$class 
label <- rep(0, length(label.text))
for(i in 1:num.label){
  class.ind <- which(label.text == level.label[i])
  label[class.ind] <- i
}
data <-as.matrix(graph.input$content[, -which(names(graph.input$content) %in% c("paper_id", "class"))])
graph.input[["features"]] <- list(data=data, label=label) 


# model establish
batch.size <- 100
random.neighbor <- c(20)
num.hidden <- c(20)
input.size <- dim(data)[2]

gcn.sym <- GCN.two.layer.model(num.hidden, num.label, dropout = 0.3)
gcn.model <- GCN.setup.model(gcn.sym,
                             random.neighbor,
                             input.size,
                             batch.size,
                             mx.ctx.default(),
                             mx.init.uniform(0.01))

# training process
num.nodes.train <- floor((length(label) * 0.1)/batch.size)*batch.size
num.nodes.valid <- floor((length(label) * 0.1)/batch.size)*batch.size
nodes.pool <- sample(c(1:length(label)), (num.nodes.train+num.nodes.valid), replace=FALSE)
nodes.train.pool <- sort(nodes.pool[1:num.nodes.train])
nodes.valid.pool <- sort(nodes.pool[(num.nodes.train+1):(num.nodes.train+num.nodes.valid)])

learning.rate <- 0.01
weight.decay <- 0
clip.gradient <- 1
optimizer <- 'sgd'
lr.scheduler <- mx.lr_scheduler.FactorScheduler(step = 50, factor=0.5, stop_factor_lr = 0.01)
gcn.model.trained <- GCN.trian.model(model = gcn.model,
                                     graph.input = graph.input,
                                     nodes.train.pool = nodes.train.pool,
                                     nodes.valid.pool = nodes.valid.pool,
                                     num.epoch = 100,
                                     learning.rate = learning.rate,
                                     weight.decay = weight.decay,
                                     clip.gradient = clip.gradient,
                                     optimizer = optimizer,
                                     lr.scheduler = lr.scheduler)