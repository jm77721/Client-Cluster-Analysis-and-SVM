os <- read.csv("OnlineSales.csv",
               stringsAsFactors = FALSE)
## Packages Used
library(ggplot2) 
library(caret) 
library(cluster)
library(factoextra)
library(fpc)
library(NeuralNetTools)
library(e1071)
library(MLeval)
library(kernlab)
library("randomForest")
load("Clus_Plot_Fns.RData")

## Preprocessing 
unique(os$SpecialDay)
ord <- os[,"SpecialDay"]
os[,"SpecialDay"] <- factor(x = os[,"SpecialDay"], 
              levels = c("0.0","0.2", "0.4", "0.6", "0.8", "1.0"),
              ordered = TRUE)
fac <- c("Month","OperatingSystems","Browser","Region","TrafficType",
    "VisitorType","Weekend","Revenue")
os[,fac] <- lapply(X = os[ ,fac], 
            FUN = factor)
ord
num <- c("Administrative","Administrative_Duration","Informational","Informational_Duration",
         "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues")
str(os)

vars <- c(num[1:9], ord,fac[1:5],fac[7]) ## Omitting VisitorType

## Normalizing
os_yj <- preProcess(x = os,
                 method = c("YeoJohnson","center","scale"))

os_yjsc <- predict(object = os_yj,
                        newdata = os)

## Outliers
outs <- sapply(os_yjsc[,num], function(x) which(abs(x) > 3))#identify outliers
outs

os_yjsc[unique(unlist(outs)),]#views outliers

os_yjsc <- os_yjsc[-(unique(unlist(outs))),]   ##Removing outliers from normalization
os <- os[-(unique(unlist(outs))),]             ##Removing outliers from original data set
nrow(os_yjsc)

## Hierarchical Clustering
daisy(x = os[,c(1:16,17)], 
      metric = "gower")

symnum(x = cor(os[ ,num]), 
       corr = TRUE)

featurePlot(x = os[ ,num], 
            y = os$Revenue,
            plot = "box")

findCorrelation(x = cor(x = os[ ,num]),
                cutoff = .75,
                names = TRUE)

vars <- vars[!vars %in% c("ProductRelated_Duration","BounceRates")]
num <- num[!num %in% c("ProductRelated_Duration","BounceRates")]

hdist <- daisy(x = os[, c(1:5,8:17)], 
               metric = "gower")

summary(hdist)

## Single Linkage

os_sing <- hclust(d = hdist, 
               method = "single")

sil_plot(dist_mat = hdist,
         method = "hc",
         hc.type = "single",
         max.k = 15)     ## Gives us an optimal k of 2

wss_plot(dist_mat = hdist, 
  method = "hc", # HCA
  hc.type = "single", 
  max.k = 15)   ##Maybe an elbow at 7 if one needs to be chosen

plot(os_sing, 
     sub = NA, xlab = NA, 
     main = "Single Linkage")

rect.hclust(tree = os_sing,
            k = 7, 
            border = hcl.colors(7))

sing_clust <- cutree(tree = os_sing, 
                       k = 2)
head(sing_clust)

## Complete Linkage
os_com <- hclust(d = hdist, 
              method = "complete")

sil_plot(dist_mat = hdist,
         method = "hc",
         hc.type = "complete",
         max.k = 15)        ## Optimal k is 5

wss_plot(dist_mat = hdist, 
         method = "hc", # HCA
         hc.type = "complete", 
         max.k = 15)        ## Optimal k is 6

plot(os_com, 
     sub = NA, xlab = NA, 
     main = "Complete Linkage")

rect.hclust(tree = os_com, 
            k = 5, 
            border = hcl.colors(5))

comp_clust <- cutree(tree = os_com, k = 5)

## Average Linkage
os_avg <- hclust(d = hdist, 
              method = "average")

sil_plot(dist_mat = hdist,
         method = "hc",
         hc.type = "average",
         max.k = 15)        ## Optimal k is 2

wss_plot(dist_mat = hdist, 
         method = "hc", # HCA
         hc.type = "average", 
         max.k = 15)        ## Optimal k is 8

plot(os_avg, 
     sub = NA, xlab = NA, 
     main = "Average Linkage")

rect.hclust(tree = os_avg, 
            k = 8, 
            border = hcl.colors(8))

avg_clust <- cutree(tree = os_avg, k = 8)

## Ward's
os_wards <- hclust(d = hdist, 
                method = "ward.D2")

sil_plot(dist_mat = hdist,
         method = "hc",
         hc.type = "ward.D2",
         max.k = 15)        ## Optimal k is 2

wss_plot(dist_mat = hdist, 
         method = "hc", # HCA
         hc.type = "ward.D2", 
         max.k = 15)        ## Optimal k is 5

plot(os_wards, 
     xlab = NA, sub = NA, 
     main = "Ward's Method")

rect.hclust(tree = os_wards, 
            k = 5, 
            border = hcl.colors(5))

wards_clust <- cutree(tree = os_wards, 
                         k = 5)

## Describing Clusters (just Ward's Method here, comparative performance evaluation below)
aggregate(x = os[ ,num], 
          by = list(wards_clust),
          FUN = mean)

aggregate(x = os[ ,c("SpecialDay",fac)], 
          by = list(wards_clust), 
          FUN = table)

table(Sales = os$VisitorType, 
      Clusters = wards_clust)

## kMeans

set.seed(607)

sil_plot(dist_mat = hdist,
         method = "kmeans",
         scaled_data = os_yjsc[,num],
         max.k = 15)           ## Optima at 5 k

wss_plot(scaled_data = os_yjsc[ ,num],
         method = "kmeans",
         max.k = 15, 
         seed_no = 831)        ## No definitive elbow. Using 5 as it seems to
                               ## product most evenly distributed clusters

os_kmeans <- kmeans(x = os_yjsc[ ,num], # data
                  centers = 5, # # of clusters
                  trace = FALSE, 
                  nstart = 30)

fviz_cluster(object = os_kmeans, 
             data = os_yjsc[,num])

matplot(t(os_kmeans$centers), # cluster centroids
        type = "l",
        ylab = "",
        xlim = c(0, 9),
        xaxt = "n",
        col = 1:4,
        lty = 1:4,
        main = "Cluster Centers")
axis(side = 1, 
     at = 1:9,
     labels = num, 
     las = 2) 
legend("left", 
       legend = 1:4,
       col = 1:4, 
       lty = 1:4, 
       cex = 0.6) 

## k-Medoids

set.seed(607)

sil_plot(dist_mat = hdist,
         method = "pam",
         scaled_data = os_yjsc[,num],
         max.k = 15)           ## Never loaded the plot

wss_plot(dist_mat = hdist,
         method = "pam",
         max.k = 15,
         seed_no = 831)     ## Never loaded the plot

os_pam <- pam(x = hdist,
            diss = TRUE,
            k = 5) 
os[os_pam$medoids, ]

clus_means_PAM <- aggregate(x = os_yjsc[ ,num], 
                            by = list(os_pam$clustering), 
                            FUN = mean)

clus_cens_PAM <- os_yjsc[os_pam$medoids,num]

matplot(t(clus_cens_PAM), 
        type = "l", 
        ylab = "", 
        xlim = c(0, 9), 
        xaxt = "n", 
        col = 1:4,
        lty = 1:4, 
        main = "Cluster Centers") 
axis(side = 1, 
     at = 1:9, 
     labels = num, 
     las = 2) 
legend("left", 
       legend = 1:4, 
       col = 1:4, 
       lty = 1:4,
       cex = 0.6)

table(Sales = os$ProductRelated, 
      Clusters = os_kmeans$cluster)

table(Sales = os$ProductRelated, 
      Clusters = os_pam$clustering)

## Adjusted Rand Index
#a. HCA

cluster.stats(d = hdist,
              clustering = sing_clust, 
              alt.clustering = as.numeric(os$VisitorType))$corrected.rand

cluster.stats(d = hdist,
              clustering = comp_clust, 
              alt.clustering = as.numeric(os$VisitorType))$corrected.rand

cluster.stats(d = hdist,
              clustering = avg_clust, 
              alt.clustering = as.numeric(os$VisitorType))$corrected.rand

cluster.stats(d = hdist,
              clustering = wards_clust, 
              alt.clustering = as.numeric(os$VisitorType))$corrected.rand

#b. K's

cluster.stats(d = dist(os_yjsc[ ,num]), # distance matrix for data used
              clustering = os_kmeans$cluster, # cluster assignments
              alt.clustering = as.numeric(os$VisitorType))$corrected.rand

cluster.stats(d = hdist, # distance matrix
              clustering = os_pam$clustering, # cluster assignments
              alt.clustering = as.numeric(os$VisitorType))$corrected.rand

## Alt Validation Methods
#HCA
sing_stat <- cluster.stats(d = hdist, 
                           clustering = sing_clust)
sing_stat$dunn

comp_stat <- cluster.stats(d = hdist, 
                           clustering = comp_clust)
comp_stat$dunn

avg_stat <- cluster.stats(d = hdist, 
                           clustering = avg_clust)
avg_stat$dunn

wards_stat <- cluster.stats(d = hdist, 
                          clustering = wards_clust)
wards_stat$dunn

# K's

kmeans_stat <- cluster.stats(d = hdist, 
                          clustering = os_pam$clustering)
kmeans_stat$dunn

pam_stat <- cluster.stats(d = hdist, 
                           clustering = os_pam$clustering)
pam_stat$dunn

### Performance Compared

c_stats <- c("max.diameter", "min.separation", 
             "average.between", "average.within",
             "dunn")

## HCA
# Cophenetic 
cor(x = hdist, y = cophenetic(x = os_sing))

# Complete Linkage
cor(x = hdist, y = cophenetic(x = os_com))

# Average Linkage
cor(x = hdist, y = cophenetic(x = os_avg))

# Ward's Method
cor(x = hdist, y = cophenetic(x = os_wards))

cbind(cor(x = hdist, y = cophenetic(x = os_sing)),
      cor(x = hdist, y = cophenetic(x = os_com)),
      cor(x = hdist, y = cophenetic(x = os_avg)), ##Average gives best distance measure of HCA
      cor(x = hdist, y = cophenetic(x = os_wards)))

sing_stat2 <- sing_stat[names(sing_stat) %in% c_stats]
comp_stat2 <- comp_stat[names(comp_stat) %in% c_stats]
avg_stat2 <- avg_stat[names(avg_stat) %in% c_stats]
wards_stat2 <- wards_stat[names(wards_stat) %in% c_stats]

cbind(sing_stat2$average.between,  ##gives highest average between clusters
      comp_stat2$average.between,
      avg_stat2$average.between,   ## runner up
      wards_stat2$average.between)

cbind(sing_stat2$min.separation,  
      comp_stat2$min.separation,  ## Lowest min separation
      avg_stat2$min.separation,   ## runner up
      wards_stat2$min.separation)

# Analysis will be using Ward's distance measure

#### SVM
## Preprocessing

os ## Outliers and other preprocessing handled from earlier in cluster analysis
os$SpecialDay
str(os)
os2<- os ## to further manipulate data as needed for SVM
os2$SpecialDay <- as.numeric(os2$SpecialDay)

os2$SpecialDay <- as.numeric(os2$SpecialDay)

str(os2)

os2$SpecialDay[is.na(os2$SpecialDay)] <- 0


os2[,c("Weekend", "Revenue")] <- lapply(X = os2[ ,c("Weekend", "Revenue")], 
                                                         FUN = class2ind, 
                                                         drop2nd = TRUE)
os2[,c("Weekend", "Revenue")]<- lapply(X = os2[ ,c("Weekend", "Revenue")], 
                                       FUN = factor)






cats <- dummyVars(formula =  ~ Month+OperatingSystems+Browser+Region
                  +TrafficType+VisitorType,
                  data = os2)
cats_dums <- predict(object = cats, 
                     newdata = os2) 

os2_dum <- data.frame(os2[ ,!names(os2) %in% c("Month","OperatingSystems",
                  "Browser","Region","TrafficType","VisitorType")],cats_dums)
levels(os2_dum$Revenue) <- c("Purchase", "No.Purchase")
plot(os2_dum$Revenue)

## Missing and Outlier Check
any(is.na(os2_dum))    ## No missing data

## Rescaling and Standardization
    ## Rescaling of numberic variables will be done during modeling

## Partitioning Data

set.seed(607)

sub <- createDataPartition(y = os2_dum$Revenue, 
                           p = 0.7, # .7 /.3 split
                           list = FALSE)

train2 <- os2_dum[sub, ]
test2 <- os2_dum[-sub, ]


## Model
#Linear
lin_mod <- svm(formula = Revenue ~ .,
               data = train2, 
               method = "C-classification", 
               kernel = "linear", 
               scale = TRUE)

summary(lin_mod)
lin_mod$fitted
# Radial
set.seed(607)

rad_mod <- svm(formula = Revenue ~ ., 
               data = train2, 
               method = "C-classification", 
               kernel = "radial", 
               scale = TRUE) 

summary(rad_mod)

## Training Performance
lin_mod$fitted
lin_train_conf <- confusionMatrix(data = lin_mod$fitted, 
                                  reference = train2$Revenue, 
                                  positive = "Purchase",
                                  mode = "everything"
                                  )

length(lin_mod$fitted)
length(train2$Revenue)
lin_train_conf

rad_train_conf <- confusionMatrix(data = rad_mod$fitted, 
                                  reference = train2$Revenue,
                                  positive = "Purchase",
                                  mode = "everything")
rad_train_conf

# Overall Comparison
cbind(Linear = lin_train_conf$overall,
      Radial = rad_train_conf$overall)
# Class-Level Comparison
cbind(Linear = lin_train_conf$byClass,
      Radial = rad_train_conf$byClass)

## Tuning

ctrl <- trainControl(method = "repeatedcv",
                     number = 5, 
                     repeats = 3, 
                     search = "random", 
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary) 
set.seed(607)

SVMFit <- train(form = Revenue ~ ., 
                data = train2, 
                method = "svmRadial", 
                preProcess = c("center", "scale"), 
                trControl = ctrl, 
                tuneLength = 10,
                metric = "ROC") # Takes considerable time to run, but does work

evalm(SVMFit)$roc

## Training Performance
tune.tr.preds <- predict(object = SVMFit,
                         newdata = train2)

SVM_trtune_conf <- confusionMatrix(data = tune.tr.preds,
                                   reference = train2$Revenue, 
                                   positive = "Purchase",
                                   mode = "everything")

tune.te.preds <- predict(object = SVMFit,
                         newdata = test2)

## Testing Performance
SVM_tetune_conf <- confusionMatrix(data = tune.te.preds, 
                                   reference = test2$Revenue, 
                                   positive = "Purchase",
                                   mode = "everything")
SVM_tetune_conf

## Goodness of Fit

# Overall
cbind(Training = SVM_trtune_conf$overall,
      Testing = SVM_tetune_conf$overall)

# Class-Level
cbind(Training = SVM_trtune_conf$byClass,
      Testing = SVM_tetune_conf$byClass) ## Balanced model

-----------------------------------
save.image("Final_Group1.RDATA")
