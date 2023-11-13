Happy Customers
==============================

Project Overivew
------------
In a rapidly growing delivery startup, we're actively addressing challenges to ensure customer happiness. This project involves analyzing recent survey data to understand and improve customer satisfaction. By focusing on enhancing our services, we aim to drive positive experiences for customers and contribute to the growth of the delivery sector initiatives.
Dataset involves 6 attributes and a target colum.
    X1 : my order was delivered on time
    X2 : contents of my order was as I expected
    X3 : I ordered everything I wanted to order
    X4 : I paid a good price for my order
    X5 : I am satisfied with my courier
    X6 : the app makes ordering easy for me
    Y : target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers

Exploratory Data Analysis
------------

Dataset was balanced with 51% of happy customers and 49% of unhappy customers.

Correlation between the features and target are as given below:
![image](https://github.com/fasmina-nasar/0t8aeVdJRQKBz5At/assets/110358522/bc6e2ef1-33cb-4a50-8809-ee434d2e6c39)

From pearson correlation, it is clear that X4 (I paid a good price for my order)  and X2 (contents of my order was as I expected) are the least important features to predict customer satisfaction. So,dropping these would give us acurate results.

 Models Implemented
------------

To get best classification model for the data, I've implemented GridSearchCV to search for the best hyperparameters for the model. I've also calculated f1-score, accuracy score, roc-auc score and also elapsed time in creating the model. Results are given below:
| model | f1-score |accuracy score | roc_auc_score | elapsed_time | best_params |
| --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.521739 | 0.500000 | 0.500000 | 1.836226 | {'max_depth': 5, 'n_estimators': 100} |
| GradientBoostingClassifier | 0.400000	| 0.454545 | 0.466667 | 1.447703 | {'max_depth': None, 'n_estimators': 100} |
| LogisticRegression | 0.727273 | 0.727273 | 0.733333 | 0.032820 | {'C': 1.0, 'penalty': 'l2'} |
| DecisionTreeClassifier | 0.476190 | 0.500000 | 0.508333 | 0.274594 | {'criterion': 'gini', 'max_depth': 15, 'min_sa... |
| KNeighborsClassifier | 0.583333 | 0.545455 | 0.541667 | 0.199248 | {'algorithm': 'auto', 'n_neighbors': 5, 'weigh... |
| LinearSVC | 0.620690 | 0.500000 | 0.475000 | 0.024247 | {'C': 2.0, 'loss': 'hinge', 'penalty': 'l2'} |
| RidgeClassifier | 0.727273 | 0.727273 | 0.733333 | 0.158977 | {'alpha': 1.0, 'solver': 'auto'} |
| ExtraTreesClassifier | 0.583333 | 0.545455 | 0.541667 | 53.212249 | {'criterion': 'entropy', 'max_depth': 5, 'min_... |
| AdaBoostClassifier | 0.692308 | 0.636364 | 0.625000 | 4.842035 | {'learning_rate': 1.9, 'n_estimators': 60} |

Based on results, however Logistic regression and Ridge classifier do have the same percentage of accuracy, time taken to create Logistic regression model is less than SVC. So, we can conclude Logistic regression (with parameters C=1.0 and penalty=l2) as the best model.

 Results and Conclusions
------------

After conductiong hyperparameter tuning and feature selection, The best performed model is Logistic regression with 73% of accuracy with hyperparameters c=1 and penalty=l2.

The confusion matrix is as follows:

![image](https://github.com/fasmina-nasar/0t8aeVdJRQKBz5At/assets/110358522/52844910-9f85-4221-8679-e323e45674f1)


I've calculated the feature importance using the coefficient gave me the follwing result:
![image](https://github.com/fasmina-nasar/0t8aeVdJRQKBz5At/assets/110358522/7a4a2803-8c64-4de5-b160-b37bb271ab96)


From this we could conclude that, X6 (easiness to place the order through app) doesn't have much contribution to predict happiness of a customer while Delivery of order on time is the main factor for the happiness of the customer, followed by the satisfaction with the courier.















Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
