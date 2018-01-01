#Apriori 

#Data Preprocessing 
#install.packages("arules")
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = F)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

#Training Apriori on the dataset 
rules = apriori(data = dataset, parameter = list(support = 0.004,confidence = 0.2))
#### LEt's try changing the support. What if a product is bought minimum 4 times a day
#### 4*7/4500 (4 times a day times 7 times a week divided by total number of transactions)
#Visualising the results 
inspect(sort(rules, by = 'lift')[1:10])
