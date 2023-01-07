#!/usr/bin/env Rscript

options(warn = -1)

# Load essential packages
library(mixOmics)
library(parallel)
library(nscancor)  # use the acor() to do deflation

# 0. create output directory
cat("-1. reading in the dataset")
datadir <- ".."
dt <- read.csv(paste0(datadir, ".."), header = T, row.names = 1)

cat("-2. setting parameter")
ncomp  <- 5
nsplit <- 10
nperm <- 10000
dt$Foldnum <- sample(1:nsplit, nrow(dt), replace = TRUE)

todo <- "scca_pca"
if ("scca_household" %in% todo) {
    cat("nice")
    household.eid <- read.csv(paste0(datadir, "id.household"), header = T)
    dt <- merge( x= household.eid, y = dt, by = "eid", all.x = TRUE)
}

# --------the parameters according to TODO------
index.X <- switch (
    todo,
    'scca' = c(17:69),
    'scca_household' = c(17:69),
    'scca_pca' = c(2:16)
)
index.Y <- c(98:118)   # mental health items (n = 21)

result.path <- switch (
    todo,
    'scca' = "output_scca_deflation",
    'scca_household' = "output_scca_household_deflation",
    'scca_pca' = "output_scca_pca_deflation",
)
dir.create(result.path)
setwd(result.path)

for (X.penalty in seq(0.5, 0.5, 0.1)) {
    
    for (Y.penalty in seq(0.5, 0.5, 0.1)) {
        
        outdir <- paste0(X.penalty, "_", Y.penalty)
        dir.create(outdir)
        setwd(outdir)
        cat("is doing -----: Xpenalty = ", X.penalty, "_ Ypenalty = ", Y.penalty, "\n")
        
        for (fold.test in seq(2, nsplit)) {  # define the i fold (i=1) rows as test dataset 
            
            cat("running fold.test = ", fold.test, "\n    component = ", 1, "\n")
            fold.path <- paste0('fold_', fold.test)
            dir.create(fold.path)
            setwd(fold.path)
            
            index.train = which(dt$Foldnum != fold.test)  # define the other folds (i!=1) rows as train dataset
            index.test  = which(dt$Foldnum == fold.test)
            
            train <- list()
            train$X <- scale(dt[index.train, index.X])  # Env datasets
            train$Y <- scale(dt[index.train, index.Y])  # mental health datasets
            test <- list()
            test$X <- scale(dt[index.test, index.X])
            test$Y <- scale(dt[index.test, index.Y])
            
            ## ------- running sgcca ------- 
            result.train = wrapper.sgcca(
                train,
                penalty = c(X.penalty, Y.penalty),
                ncomp = 1,
                scheme = "centroid",
                scale = TRUE
            )
            
            xt <- list() -> yt
            xt$train[[1]] <- train$X
            yt$train[[1]] <- train$Y
            xt$test[[1]]  <- test$X
            yt$test[[1]]  <- test$Y
            
            #loading <- list() -> Yloading # loading here is the same as xcoef
            weight <- list()
            weight$trainX[[1]] <- result.train$loadings$X -> weight$testX[[1]]
            weight$trainY[[1]] <- result.train$loadings$Y -> weight$testY[[1]]
            
            component <- list()
            component$trainX[[1]] <- result.train$variates$X
            component$trainY[[1]] <- result.train$variates$Y
            component$testX[[1]]  <- xt$test[[1]] %*% weight$trainX[[1]]
            component$testY[[1]]  <- yt$test[[1]] %*% weight$trainY[[1]]
            
            corlist <- list()
            corlist$train[1] <- cor(component$trainX[[1]], component$trainY[[1]])
            corlist$test[1]  <- cor(component$testX[[1]],  component$testY[[1]])
            
            # do deflation
            for (i in seq(2, ncomp)) {
                cat("    component = ", i, "\n")
                # do train
                ns.train <- acor(xt$train[[i-1]], weight$trainX[[i-1]], yt$train[[i-1]], weight$trainY[[i-1]], xscale = TRUE, yscale = TRUE)
                data.train <-list()
                xt$train[[i]] <- ns.train$xp -> data.train$X
                yt$train[[i]] <- ns.train$yp -> data.train$Y 
                
                tmp.train = wrapper.sgcca(
                    data.train,
                    penalty = c(X.penalty, Y.penalty),
                    ncomp = 1,
                    scheme = "centroid",
                    scale = TRUE
                )
                weight$trainX[[i]] <- tmp.train$loadings$X
                weight$trainY[[i]] <- tmp.train$loadings$Y
                
                component$trainX[[i]] <- tmp.train$variates$X
                component$trainY[[i]] <- tmp.train$variates$Y
                
                corlist$train[i] <- cor(component$trainX[[i]], component$trainY[[i]])
                
                # do on  test 
                # get deflated test data
                ns.test  <- acor(xt$test[[i-1]], weight$testX[[i-1]], yt$test[[i-1]], weight$testY[[i-1]], xscale = TRUE, yscale = TRUE)
                xt$test[[i]] <- ns.test$xp
                yt$test[[i]] <- ns.test$yp
                
                # use the weight from train
                weight$testX[[i]] <- weight$trainX[[i]] 
                weight$testY[[i]] <- weight$trainY[[i]] 
                
                component$testX[[i]] <- xt$test[[i]] %*% weight$testX[[i]] 
                component$testY[[i]] <- yt$test[[i]] %*% weight$testY[[i]] 
                
                corlist$test[i] <- cor(component$testX[[i]], component$testY[[i]])
            }
            cat("    saving the results -----------", "\n")
            #' weight
            write.csv(weight$trainX, "weight_trainX.csv")  # weight for train and test are the same
            write.csv(weight$trainY, "weight_trainY.csv") 
            write.csv(weight$testX,  "weight_testX.csv")
            write.csv(weight$testY,  "weight_testY.csv")
            
            #' component 
            write.csv(component$trainX, "component_trainX.csv")
            write.csv(component$trainY, "component_trainY.csv")
            write.csv(component$testX, "component_testX.csv")
            write.csv(component$testY, "component_testY.csv")
            
            #' correlation
            write.csv(corlist, "corlist.csv")
            
            #' loading and cross loading
            loading <- list() -> crossloading
            for (i in seq(1, ncomp)) {
                loading$trainX[[i]] <- cor(xt$train[[i]], component$trainX[[i]])
                loading$trainY[[i]] <- cor(yt$train[[i]], component$trainY[[i]])
                loading$testX[[i]] <- cor(xt$test[[i]], component$testX[[i]])
                loading$testY[[i]] <- cor(yt$test[[i]], component$testY[[i]])
                
                crossloading$trainX_Y.component <- cor(xt$train[[i]], component$trainY[[i]])
                crossloading$trainY_X.component <- cor(yt$train[[i]], component$trainX[[i]])
                crossloading$testX_Y.component <- cor(xt$test[[i]], component$testY[[i]])
                crossloading$testY_X.component <- cor(yt$test[[i]], component$testX[[i]])
            }
            write.csv(loading$trainX, "loading_trainX.csv")
            write.csv(loading$trainY, "loading_trainY.csv")
            write.csv(loading$testX,  "loading_testX.csv")
            write.csv(loading$testY,  "loading_testY.csv")
            
            write.csv(crossloading$trainX_Y.component, "crossloading_trainX_Y.component.csv")
            write.csv(crossloading$trainY_X.component, "crossloading_trainY_X.component.csv")
            write.csv(crossloading$testX_Y.component,  "crossloading_testX_Y.component.csv")
            write.csv(crossloading$testY_X.component,  "crossloading_testY_X.component.csv")
            
            setwd("../") 
        }
    }
    
}

## -------------- loading
meanloading_trainX = data.frame(read.csv(paste0("output/loading_trainX_", 1, ".csv"), row.names = 1))
meanloading_trainY = data.frame(read.csv(paste0("output/loading_trainY_", 1, ".csv"), row.names = 1))

for (i in seq(2, 10)) {
    meanloading_trainX <- meanloading_trainX + data.frame(read.csv(paste0("output/loading_trainX_", i, ".csv"), row.names = 1))
    meanloading_trainY <- meanloading_trainY + data.frame(read.csv(paste0("output/loading_trainY_", i, ".csv"), row.names = 1))
}
meanloading_trainX = meanloading_trainX / 10
meanloading_trainY = meanloading_trainY / 10
write.csv(meanloading_trainX, paste0("output/meanloading_trainX", ".csv"))
write.csv(meanloading_trainY, paste0("output/meanloading_trainY", ".csv"))


## -------------- cross loading
mean_crossloading.train_env_mh.comp <- matrix(0, 5, 53)
mean_crossloading.train_mh_env.comp <- matrix(0, 5, 21)
for (i in seq(1, 10)) {
    cat(i)
    component.trainX  <- data.frame(read.csv(paste0("output/component_trainX_", i, ".csv"), header = T, row.names = 1))
    component.trainY  <- data.frame(read.csv(paste0("output/component_trainY_", i, ".csv"), header = T, row.names = 1))
    
    index.train = which(dt$Foldnum != i) 
    
    crossloading.train_env_mh.comp <- cor(component.trainY, dt[index.train, index.X])
    crossloading.train_mh_env.comp <- cor(component.trainX, dt[index.train, index.Y])
    mean_crossloading.train_env_mh.comp <- mean_crossloading.train_env_mh.comp + crossloading.train_env_mh.comp
    mean_crossloading.train_mh_env.comp <- mean_crossloading.train_mh_env.comp + crossloading.train_mh_env.comp
    
    write.csv(crossloading.train_env_mh.comp, paste0("output/crossloading.train_env_mh.comp_", i, ".csv"))
    write.csv(crossloading.train_mh_env.comp, paste0("output/crossloading.train_mh_env.comp_", i, ".csv"))
}

mean_crossloading.train_env_mh.comp <- mean_crossloading.train_env_mh.comp / 10
mean_crossloading.train_mh_env.comp <- mean_crossloading.train_mh_env.comp / 10
write.csv(mean_crossloading.train_env_mh.comp, paste0("output/mean_crossloading.train_env_mh.comp_", ".csv"))
write.csv(mean_crossloading.train_mh_env.comp, paste0("output/mean_crossloading.train_mh_env.comp_", ".csv"))


## --------------------------------- bootstrap resampling stability ---------------------------------------
cat(">>>>-------- bootstrap resampling stability", "\n")
set.seed(123)
perm <- 1:nperm
boot_func <- function(i) {
    cor_boot_struct <- matrix(0, 15, 1)
    sampleseq <- seq(0.1, 1.5, 0.1)
    for (j in sampleseq) {
        trainsamp <- sample(nrow(train.sparsed$X), size = ceiling(j*nrow(train.sparsed$X)), replace = TRUE)
        train.list = list()
        train.list$X = train.sparsed$X[trainsamp, ]
        train.list$Y = train.sparsed$Y[trainsamp, ]
        tmp.rgcca = wrapper.sgcca(train.list,
                                  penalty = c(1, 1),
                                  ncomp = 1, scheme="centroid")
        cor_boot_struct[which(sampleseq==j), ] <- diag(cor(tmp.rgcca$variates$X, tmp.rgcca$variates$Y))
    }
    return(cor_boot_struct)
}

results.4 <- mclapply(perm, boot_func, mc.cores = 1)
absresults <- lapply(results.4, abs)
struct_resample <- t(matrix(unlist(absresults), 15, length(perm)))
write.csv(struct_resample, "struct_resample.csv")


