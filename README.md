# RelsB

First, we addressed all missing values in the dataset.

We thoroughly examined each missing value to determine whether it represented an actual missing value or if it simply indicated the absence of data for that particular variable.

For categorical variables, we assigned the value 'unknown' where appropriate.

For numerical variables, such as latitude and longitude, we used the postal code to group the data into clusters and calculated the central value (mean) within each group to fill in the missing values.

If there were a variable with a lot of missings, such as the StreetDirectionSuffix,we join the value with StreetDirectionPrefix in order to avoid to remove this variable. 

We also decided to remove some values such as the Postal Code Plus 4 since it has a lot of missings and we can obtain the same information by using the Postal Code. 

For other variables such as the parking features, we changed to binary, indicating if the house has or not a parking. 

For some numerical variables such as C and Q, we used MICE to imputed the values.  

We also decided to create new variables such as the age of the building instead of the creation year. And we used that prediction to imputed the value of NewConstructionYN. 

For HighSchoolDistrict we found the most nearest house and we imputed with the same values. 

For the variables which contained a list, we did one hot encoding and we calculated the correlation with the target in order to find the importance one. 

This is a summarize of the pre-processing of the missing but you can see the complete  justifications and modifications. 


After that we proceed to calculate the univariate and the bivariant to see the correlations. We used plots to see the distribution of the data, we use PCA to see the correlation of the data and we used the Isolation Forest to see the outliers. We also decided to joint the most correlation data such us the Q and C values. 

In order to create the models we started by creating a linear regression to see the correlation of each variable with the target. After that, we used it as a baseline of out problem. 

After that, we used several methods: 

EBM
XBoosting
Random Forest
Neuronal Networks 
LGBM regressor

For each method, we studied the correlation data and after that, we tried to change the characteristic used in order to see the most importance one. We did it with feature importance. 

We also tried with other data such us with outliers (it worked better). 

We also used KRIGING in order to add knowledge. And we also used latitude and longitude to study the data above the map. Finally, we search for new data on internet. We used the latitude and longitude information to find near shops, schools and interesting points. However, we did not have enough time to use it. 

We thought that a good approach would be to add some information of climate change. 

We want to add that we used MLFLOW in order to track the metrics of out models and SHAP to add explainability.   
