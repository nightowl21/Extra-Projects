# About

The competition was hosted on Kaggle and the data was graciously provided by Yummly. Yummly is a well known website that provides recipe recommendations personlized to user's taste. The training and test data are provided as json blobs. Each observation in the training data represents ID, cuisine type and ingredient list for a recipe. In total there are ~50,000 recipes in the training dataset. The test data has recipe ID and ingredient list for 20,000 recipies. The challenge is to predict the cuisine for these recipes given the ingredient list.

https://www.kaggle.com/c/whats-cooking

I worked on this competition as a final project for my Machine Learning Class at my Master's program. I chose this competition because:

1. The dataset was very interesting.
2. The problem is easy to understand.
3. It gave me the opportunity to apply NLP and machine learning skills.
4. It is a practical business problem.

It was a lot of fun to work on this project. I speacially loved the feature engineering part. This project was the first opportunity where I got to apply ensemble modelling and see how powerfull it is. Individual models gave a lower accuracy than the final ensembled model. This model got me a rank of 108 (amongst top 10% participants) in the competition. https://www.kaggle.com/mlgeek21



## Approach

I used python for this project as NLP data handling and processing capabilties are better handled in python. I did some EDA to understand the data. I perfomed a lot of Data Cleaning and Manipulation to form the design matrix. Then I applied a couple of machine learning algorithms, used GridSearchCV to determine best hyper parameters. Finally, I ensembled these models to get better accuracy.

#### Data Cleaning and Preperation  

1. I started by exploring the data. This is a multiclass classification problem with 20 classes. Since this problem  sufferes from class imbalance, it made sense to balance the data. However, I decided not to do so for 2 reasons. 

* The size of training data is small
* The aim of the problem is to achive high accuracy. 


2. I started with EDA to understand which ingredients are common in various cuisines (eg, salt, sugar, oil etc.). Since such ingredients are foundational in cooking, they can be removed without much loss of information.

3. As with any NLP problem, I removed stop words and also stemmed the words. I removed punctuations and number and also words that detone the quantity or amount of an ingredient. I think that the quantity of an ingredient does not speak a lot to the what the recipe is, as the quantity may depend on the how many servings the recipe makes. 

4. After the above steps, I had about 6,000 unique ingredients. I decided to look to through them to understand what kind of ingredients are being used. I noticed that ingredient list had brand names, like heinz, campbell etc. I decided to remove them as well.

5. Looking through the list of ingredients, I also noticed many different spellings of the same ingredient. Since Yummly is a free text platform, these things can be easily missed. I looked at autocorrecting these spellings using levenshtine distance, but it resulted in more false positives and therefore introduced errors. Since, the list of ingredients was not massive, I decided to correct the spellings by mannually.

6. After this I used frequency of the ingredients to form the data design matrix for the problem. I tried TFID instead of frequency counts as well, but that did give good accuracy and hence the final model uses frequency counts instead.

#### Modelling

1. Model Selection: 3 models were chosen namely XGB, LR and RF. This choice was made as the loss functions are different for these models and hence together they can learn different aspects of the data.

2. Cross Validation: Cross validation with Grid Search was done to pick the right hyper parameters. To avoid over tuning and also long training times, I used RandomGrid search with CV.

3. Ensembling: The models were ensembled using simple aggregattion. The model was trained on a subset of original training data and performance was noted on the validation set (30% of total). This helped determine the weights to be used to aggregate the models.

## Scripts

1. `data/` rRepo that contains test and train data in the form of json blobs. 
2. `data_processing.py` Processes and cleans the data. Removes words and corrects spellings of the words.
3. `cross_validate.py` Runs cross validation using RandomGridCV functionality in sklearn.model_selection. 3 Models used are XGB, LR and RF.
4. `ensembled_model.py` Uses a subset of training data to train the model. The validation set performance is used to finalize the weights for each model. The class with the highest weighted average probabilty is assigned as the final prediction for the observation.
5. `final_model.py` Using the model parameters from `cross_validate.py` and model weights from `final_model.py`, final model is trained on the entire training data and predictions are made and weighted average is taken on the test set. These predictions were then submited to Kaggle.






