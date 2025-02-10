# Data Science Midterm Project

## Project/Goals

The goal of this project is to develop a model to predict the selling price of a house based the house's features as well as the demographic region of the house. 

## Process
### Step 1: Importing Datasets
- The first step in this process was to compile all of the data from regional JSONs into a single csv file. The next step was to add additional regional data based on the location of the house. We imported cost of living information, population information and US city information. We merged all of these based on the postal code and state of the house.
### Step 2: Cleaning Data
- The next step was to clean the data. We removed the null values from the Sold Price columns. We also filled Null feature information with zeros. Additionally, we removed features with large amounts of null values or ones that we realised were not useful. 
### Step 3: Feature Selection
- After cleaning the initial data. The final step before model testing was to minimize the number of columns, to improve model performance and computational testing, we needed to minimize the number of features. There were over 100 columns in this dataset with varying levels of multicolinearity and relevance to the dataset and the model.
### Step 4: Model Testing & Evaluation
- The next step in our process was to test our dataset across various models. We used Linear regression with Lasso and Ridge, several versions of SVR, a random forest regression as well as two XG boost models. We iterated through them with cross-validation to find the best scores possible for each model. We found that after testing the various models that the Random Forest Model had the highest score. 
### Step 5: Building Pipeline
- The first step was definining each function of our pipeline. The first was a cross-validation function, which would split our dataset into an identified number of folds. The next was a hyperparameter function, the purpose of that is to iterate through a parameter grid to find the best parameters for our dataset and model. The next two are to fit a model with the best parameters and to finally make a prediction. 


## Results
Our top performing model was a Random Forest Regressor. Based on the data that was available to us, this model performed the best in perdicting the variance in sold price of houses. The R-squared score was 0.59, which means that our model was most reliable for explaining roughly 59% of the variance in the data. This means that our model was moderately successful in capturing the relationship between the selling price of a house and various factors. This can be useful for price estimation or market analysis. 

## Challenges 
1.) The first challenge was the large amount of null values in the dataset. Many columns were null or had to be dropped. While, the data was still useable, this impact the level of detail and information that we were able to obtain from the dataset.The feature set also contained a number of null values. While features that were not present were zero, we had to make assumptions with the features in order to keep the information available. 
2.) The second challenge was the way the dataset was structured. The initial dataset was structured with over 300 columns. This meant that we were limited to using only a few features to train our model or having to run many different models and feature selections to find the best one. The columns were also extremely redundant, leading to many of the columns having strong multicolinearity. We had to sift through that data and clean it up prior to being able to train our model effectively. 
3.) Imbalanced dataset. The dataset was imbalanced with the number of houses being sold vs being listed. This represents a chalenge for us to be able to accurately train our model on the final selling price of a house as it is not representative of all housing data in the region.


## Future Goals
- Collect more data on the houses. The more data we have, the better our model will be at predicting the selling price of a house. This includes more information on the features of the house, as well as more information on the regional data. 
- Use a differnt model. The model that we used was a baseline model and it was not the best model for the task. With more time, we could explore other models that would be better suited for the task. 
- Build an unsupervised learning model to find trends in housing data outside of the ones predicted in this model. 
- Use a different way of dealing with the imbalanced dataset. We used a random sample of the data to make the dataset more balanced. However, this is not the best way of dealing with an imbalanced dataset and there are other methods that could be used. 
- Use a different way of dealing with the multicolinearity. We used a PCA to reduce the dimensionality of the data. However, this is not the best way of dealing with multicolinearity and there are other methods that could be used. 

