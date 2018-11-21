require(Matrix)
require(RSpectra)
require(rjson)
require(igraph)

normalise.adj <- function(adj){
  indicies <- which(adj!=0, arr.ind = TRUE)
  i <- indicies[,1]
  j <- indicies[,2]
  x <- rep(1, length(i))
  adj <- sparseMatrix(x=x, i=i, j=j, dims=dim(adj))
  adj <- adj + Diagonal(dim(adj)[1])
  rowsum <- rowSums(adj,dims = 1,sparseResult = FALSE)
  d_inv_sqrt <- sqrt(rowsum)
  d_inv_sqrt[is.infinite(d_inv_sqrt)] <- 0
  d_mat_inv_sqrt<- Diagonal(length(d_inv_sqrt), x=d_inv_sqrt)
  A <- d_mat_inv_sqrt%*%adj%*%d_mat_inv_sqrt
  return(A)
}

chebyshev.polynomials <- function(adj, k){
  adj.shape <- dim(adj)
  adj.normalised <- normalise.adj(adj)
  laplacian <- Diagonal(adj.shape[1]) - adj.normalised
  largest_eigval <- eigen(laplacian,only.values = TRUE)$values
  scaled_laplacian <- (2/largest_eigval[1])*laplacian - Diagonal(adj.shape[1])
  t_k <-c(Diagonal(adj.shape[1]), scaled_laplacian)
  for(i in 3:k){
    t_k_minus_one <- t_k[[length(t_k)]]
    t_k_minus_two <- t_k[[(length(t_k)-1)]]
    t_k_appender <- 2*scaled_laplacian%*%t_k_minus_one - t_k_minus_two
    t_k <- c(t_k, t_k_appender)
  }
  return(t_k)
}

Graph.receptive.fields.computation <- function(batch.begin,
                                               batch.size,
                                               P, 
                                               support, 
                                               adj,
                                               K,
                                               random.neighbor,
                                               layer.P = NULL,
                                               layer.tP = NULL)
{
  layer.tP <- list()
  layer.H <- list(c(batch.begin:(batch.begin+batch.size-1)))
  dim2 <- 0
  for(ly in 1:K){
    # determin the dimensions of P matrix
    if(ly == 1){
      dim1 <- batch.size
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
      neighbor.index  <- setdiff(neighbor.index, c(node))
      neighbor.number <- length(neighbor.index)
      if(neighbor.number <= (random.neighbor[ly]-1)){
        neighbor.vec <- neighbor.index
      }else{
        sample.neighbor.index <- sample(neighbor.number, (random.neighbor[ly]-1))
        neighbor.vec <- neighbor.index[sample.neighbor.index]
      }
      neighbor.vec <- c(neighbor.vec, node)
      node.stats <- list(node_index = node, neighbor_num = neighbor.number, neighbor_vec = neighbor.vec)
      P.neighbor[[count]] <- node.stats
      count <- count + 1
      all.vectices <- c(all.vectices, neighbor.vec)
    }  
    sort.unique.vectices <- sort(unique(all.vectices))
    
    # determine P matrix
    for(ti in 1:length(P.neighbor)){
      node.info <- P.neighbor[[ti]]
      node.neighbor.indcies <- match(node.info$neighbor_vec, sort.unique.vectices)
      for(j in 1:length(node.info$neighbor_vec)){
        vi <- node.info$node_index
        vj <- node.info$neighbor_vec[j]
        tj <- node.neighbor.indcies[j]
        if(node.info$neighbor_num <= (random.neighbor[ly]-1)){
          d <- node.info$neighbor_num
        }else{
          d <- (random.neighbor[ly]-1)
        }
        P.tilde[ti,tj] <- support[vi,vj]*(node.info$neighbor_num/d)
      }
    }
    layer.tP[[ly]] <-P.tilde
    layer.H[[ly+1]] <- sort.unique.vectices
  }
  outputs <- list(tP=layer.tP, H=layer.H)
  return(outputs)
}

loaddata.ppi <- function(){
  json_graph <- "./example_data/PPI/toy-ppi-G.json"
  G_data  <- fromJSON(paste(readLines(json_graph), collapse=""))
  edges <- matrix(unlist(G_data$links), ncol = 4, byrow = TRUE)[,3:4]+1
  graph <- graph_from_edgelist(edges, directed = FALSE)
  adjmatrix <- as_adj(graph, type = 'both', sparse = igraph_opt("sparsematrices"))
  graph2 <- graph_from_adjacency_matrix(adjmatrix, mode='undirected', diag=FALSE)
  
  json_class <- "./example_data/PPI/toy-ppi-class_map.json"
  G_class  <- fromJSON(paste(readLines(json_class), collapse=""))
  
  json_idmap <- "./example_data/PPI/toy-ppi-id_map.json"
  G_idmap  <- fromJSON(paste(readLines(json_idmap), collapse=""))
  
  csv_feats <- "./example_data/PPI/toy-ppi-feats.csv"
  feats <- read.csv(csv_feats, header = FALSE)
  
  D.sqrt <- sqrt(colSums(adjmatrix))
  
  A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
  
  P <- D.sqrt%*%A.tilde%*%D.sqrt
  
  outputs <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, features = feats, graph = graph, class = G_class)
  return(outputs)
}

loaddata.cora <- function(){
  csv_cites <- "./example_data/CORA/cites.csv"
  edges.cites <- read.csv(csv_cites, header = FALSE)
  edges.cites <- as.matrix(edges.cites[2:dim(edges.cites)[1],])
  
  csv_paper <- "./example_data/CORA/paper.csv"
  paper.class <- read.csv(csv_paper, header = FALSE)
  paper.class <- as.matrix(paper.class[2:dim(paper.class)[1],])
  
  for(i in 1:dim(paper.class)[1]){
    inds.v1 <- which(edges.cites[,1] == paper.class[i,1])
    inds.v2 <- which(edges.cites[,2] == paper.class[i,1])
    
    edges.cites[inds.v1,1] <- i
    edges.cites[inds.v2,2] <- i
  }
  edges.cites<-apply(edges.cites, 2, as.numeric)
  class(edges.cites) <- "numeric"
  storage.mode(edges.cites) <- "numeric"
  
  graph <- graph_from_edgelist(edges.cites, directed = FALSE)
  adjmatrix <- as_adj(graph, type = 'both', sparse = igraph_opt("sparsematrices"))
  
  D.sqrt <- sqrt(colSums(adjmatrix))
  
  A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
  
  P <- D.sqrt%*%A.tilde%*%D.sqrt
  
  outputs <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, graph = graph, class = paper.class)
  return(outputs)
}