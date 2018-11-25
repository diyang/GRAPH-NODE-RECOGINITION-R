require(Matrix)
require(RSpectra)
require(rjson)
require(igraph)

Graph.receptive.fields.computation <- function(nodes.pool,
                                               P, 
                                               adj,
                                               random.neighbor)
{
  layer.tP <- list()
  nodes.sort <- sort(nodes.pool)
  layer.H <- list(nodes.sort)
  included.node <- layer.H[[1]]
  K <- length(random.neighbor)
  node.size <- length(nodes.pool)
  
  for(ly in 1:K){
    # determin the dimensions of P matrix
    if(ly == 1){
      dim1 <- node.size
    }else{
      dim1 <- dim2
    }
    dim2 <- dim1*(random.neighbor[ly])
    P.tilde <- matrix(0, dim1, dim2)
    
    # define container
    all.vectices <- c()
    P.neighbor <- list()
    
    # determine neighborhood
    count <- 1
    for(node in layer.H[[ly]]){
      neighbor.index  <- which(adj[node,] > 0)
      neighbor.number <- length(neighbor.index)
      neighbor.index.diff  <- setdiff(neighbor.index, included.node)
      neighbor.diff.number <- length(neighbor.index.diff)
      
      #neighbor.index  <- setdiff(neighbor.index, c(node))
      if(neighbor.diff.number == 0){
          neighbor.vec <- NULL
      }else{
        if(neighbor.diff.number <= (random.neighbor[ly])){
          neighbor.vec <- neighbor.index.diff
        }else{
          sample.neighbor.index <- sample(neighbor.diff.number, random.neighbor[ly], replace=FALSE)
          neighbor.vec <- neighbor.index.diff[sample.neighbor.index]
        }
        all.vectices <- c(all.vectices, neighbor.vec)
      }
      node.stats <- list(node_index = node, neighbor_num = neighbor.number, neighbor_vec = neighbor.vec)
      P.neighbor[[count]] <- node.stats
      count <- count + 1
    }  
    sort.unique.vectices <- sort(unique(all.vectices))
    included.node <- c(included.node, sort.unique.vectices)
    
    # determine P matrix
    for(ti in 1:length(P.neighbor)){
      node.info <- P.neighbor[[ti]]
      if(!is.null(node.info$neighbor_vec)){
        node.neighbor.indcies <- match(node.info$neighbor_vec, sort.unique.vectices)
        for(j in 1:length(node.info$neighbor_vec)){
          vi <- node.info$node_index
          vj <- node.info$neighbor_vec[j]
          tj <- node.neighbor.indcies[j]
          if(length(node.info$neighbor_vec) <= random.neighbor[ly]){
            d <- length(node.info$neighbor_vec)
          }else{
            d <- random.neighbor[ly]
          }
          P.tilde[ti,tj] <- P[vi,vj]*(node.info$neighbor_num/d)
        }
      }
    }
    layer.tP[[ly]] <-P.tilde
    layer.H[[ly+1]] <- sort.unique.vectices
  }
  outputs <- list(tP=layer.tP, H=layer.H)
  return(outputs)
}

loaddata.ppi <- function(){
  json_graph <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-G.json"
  #json_graph <- "./example_data/PPI/toy-ppi-G.json"
  G_data  <- fromJSON(paste(readLines(json_graph), collapse=""))
  edges <- matrix(unlist(G_data$links), ncol = 4, byrow = TRUE)[,3:4]+1
  graph <- graph_from_edgelist(edges, directed = FALSE)
  adjmatrix <- as_adj(graph, type = 'both', sparse = igraph_opt("sparsematrices"))
  #graph2 <- graph_from_adjacency_matrix(adjmatrix, mode='undirected', diag=FALSE)
  
  json_class <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-class_map.json"
  #json_class <- "./example_data/PPI/toy-ppi-class_map.json"
  G_class  <- fromJSON(paste(readLines(json_class), collapse=""))
  
  json_idmap <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-id_map.json"
  #json_idmap <- "./example_data/PPI/toy-ppi-id_map.json"
  G_idmap  <- fromJSON(paste(readLines(json_idmap), collapse=""))
  
  csv_feats <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-feats.csv"
  #csv_feats <- "./example_data/PPI/toy-ppi-feats.csv"
  feats <- read.csv(csv_feats, header = FALSE)
  
  D.sqrt <- sqrt(colSums(adjmatrix))
  
  A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
  
  P <- diag(D.sqrt)%*%A.tilde%*%diag(D.sqrt)
  
  outputs <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, features = feats, graph = graph, class = G_class)
  return(outputs)
}

loaddata.cora <- function(){
  #csv_cites <-   "I:/Desktop/R/SAGE-GRAPH-R/example_data/CORA/cites.csv"
  csv_cites <- "./example_data/CORA/cites.csv"
  edges.cites <- read.csv(csv_cites, header = FALSE)
  edges.cites <- as.matrix(edges.cites[2:dim(edges.cites)[1],])
  
  #csv_paper <-   "I:/Desktop/R/SAGE-GRAPH-R/example_data/CORA/paper.csv"
  csv_paper <- "./example_data/CORA/paper.csv"
  paper.class <- read.csv(csv_paper, header = FALSE)
  paper.class <- as.matrix(paper.class[2:dim(paper.class)[1],])
  
  #csv_content <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/CORA/content.csv"
  csv_content <- "./example_data/CORA/content.csv"
  content.class <- read.csv(csv_content, header = FALSE)
  content.class <- content.class[2:dim(content.class)[1],]
  column.names <-  c("paper_id",as.character(unique(content.class$V2)),"class")
  num.cols <- length(column.names)
  num.paper <- dim(paper.class)[1]
  
  content.df <- data.frame(matrix(0, ncol = num.cols, nrow = num.paper))
  colnames(content.df) <- column.names
  
  for(i in 1:dim(paper.class)[1]){
    inds.v1 <- which(edges.cites[,1] == paper.class[i,1])
    inds.v2 <- which(edges.cites[,2] == paper.class[i,1])
    
    edges.cites[inds.v1,1] <- i
    edges.cites[inds.v2,2] <- i
    
    inds.paper_id <-which(content.class[,1] == paper.class[i,1])
    cite.ids <- content.class[inds.paper_id,2]
    content.df[i,1] <- paper.class[i,1]
    content.df[i,num.cols] <- paper.class[i,2]
    for(j in 1:length(cite.ids)){
      ind.class <- which(column.names == cite.ids[j])
      content.df[i,ind.class] <- 1 
    }
  }
  edges.cites<-apply(edges.cites, 2, as.numeric)
  class(edges.cites) <- "numeric"
  storage.mode(edges.cites) <- "numeric"
  
  graph <- graph_from_edgelist(edges.cites, directed = FALSE)
  adjmatrix <- as_adj(graph, type = 'both', sparse = igraph_opt("sparsematrices"))
  
  D.sqrt <- sqrt(colSums(adjmatrix))
  
  A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
  
  P <- diag(D.sqrt)%*%A.tilde%*%diag(D.sqrt)
  
  outputs <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, graph = graph, content = content.df)
  return(outputs)
}