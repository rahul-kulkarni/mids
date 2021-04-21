Introduction
In the online advertising industry, one of the greatest challenges is to develop systems that effectively serve ads to their intended audience. Targeted advertisement systems use vast amounts of data about user behavior and preferences to determine which ads are most relevant to each user. The task of predicting advertisement click-throughs requires a solution that is (1) fast enough to handle the real-time nature of serving ads, (2) predictive enough to select ads that users will click on, and (3) scalable enough to handle vasts amounts of user data for model training and prediction.

Problem Statement
For this project, we’ll be exploring the targeted advertising space by developing our own supervised model to predict advertisement click-through rate (CTR). We’ll be using a large advertising dataset by Criteo that includes several anonymized fields related to the user, the webpage, and the advertisement being served at a given time. Using this data, our goal is to develop a supervised model to accurately and scalably predict the likelihood a user will click on a given ad.

This problem can be described as a probabilistic binary classification problem where the predicted output ($\hat{Y}$) is the probability that the user will click on the ad served to them. The response variable ($Y$) in our dataset is a binary variable encoded to 0 and 1 representing no-click and click, respectively. To evaluate our model’s accuracy, we will use a logarithmic loss function to compare the predicted probability that a user clicks on a given ad compared to the actual response variable (click or no-click). The logarithmic loss is averaged across all true values ($y_i$) and predicted probabilities ($\hat{y}_i$) like so:

$$LogLoss = - \frac{1}{N} \Sigma_{i=1}^N (y_i log(\hat{y}_i) + (1-y_i)log(1-\hat{y}_i))$$

