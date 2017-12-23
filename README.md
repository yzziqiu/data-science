# Data Science
To record the learning process for data science, including sentiment analysis and machine learning

### Few throwbacks

**Main difference** between 2.x and 3.x

- except Exception,e: -> except Exception as e:

- print must include brackets i.e. print()
- raw_input() -> input()

- do not need decode('gbk')

### Python Basic
For detailed learning, please check [python-class repo](https://github.com/yzziqiu/python-class)

## Projects using library

### 2017-2018 NBA Games Prediction
Given data results, opponents and game data from NBA stats website, I use cross validation and linear model. Training dataset with Logistic Regression, we can know the probability of winning the game between specific teams.

### Douban Movie Reviews Extraction
Along with Jupyter Notebook, I use few python library to scrape data of now-playing movies from Douban. urllib and BeautifulSoup might be curcial for scraping website and extracting useful information. Besides that, I use re for cleaning data, jieba for lexicon, pandas and numpy for calculating frequecy of every single word. Finally, using matplotlib and wordcloud for making the graph.

### Stock
Scraping information from Baidu stock and eastmoney, list all the potential rising stocks.

### qichacha
Practice of scraping useful information from Enterprise info website Qichacha and making a spreadsheet.

## TensorFlow

Few practice with beginners of TensorFlow
1. Introduction of TensorFlow

Building a linear model and generating a graph.

2. MNIST Application (with input_data.py)
- softmax

accuracy 92%

3. Regression models

Logistic Regression and Linear Regression

4. RNN

5. Auto text generation

LSTM and RNN

- CNN(advanced)

accuracy 97%

## Sentiment Analysis

[data download from Kaggle](https://github.com/yzziqiu/data-science/blob/master/sentiment-analysis/Tweets.csv)

- Twitter Sentiment of U.S. airlines

Discard all neutral words, using regression. Accuracy on training and validation.

- Twitter Prediction of U.S. airlines

Drawing mood counting graph, which reveals the comparison and negative reasons of different airlines. Comparison performance among different classifiers: LogisticRegression, KNeighborsClassifier
    SVC, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, GaussianNB


## Kaggle

1. Amazon reviews text generation

TensorFlow, RNN and LSTM

2. Data science and Machine Learning beginners

- matplotlib, pandas

- sklearn,

Supervised: KNN, CV

Unsupervised: Kmeans, TSNE, PCA

3. feature ranking

## Machine Learning

1. deep learning

2. (linear regression) house price Prediction

3. (classifiers) sentiment analysis
