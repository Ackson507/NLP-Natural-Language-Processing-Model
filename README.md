# Natural-Language-Processing-Model
Here, we will implement these steps to build a language model in NLP. NLP focuses on the interaction between computers and human language, enabling machines to understand, interpret, and generate human language in a way that is both meaningful and useful.
![1_HgXA9v1EsqlrRDaC_iORhQ](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/315f724f-ab31-468b-b083-bdd931ca6061)



### Overview
This project aims to develop a Natural Language Processing (NLP) model for classifying patient reviews to determine the type of disease or medical condition they are experiencing. By analyzing patient reviews, healthcare providers can gain insights into the prevalence of various diseases or conditions among their patient population, which can inform treatment strategies and overall patient care.

### Problem Statement
Problem Statement: Develop an NLP model to classify patient reviews and identify the type of disease or medical condition mentioned in the reviews.

### Objective
Objective: The primary objective is to accurately classify patient reviews to determine the underlying disease or medical condition, facilitating proactive healthcare management and targeted interventions.

### Data Collection and Loading
We have sourced a compiled list of a dataset from an health sector which collect patient reviews from various sources such as online review platforms, healthcare forums, patient surveys, or electronic health records (EHRs).
```python
# Libraries
# Data manipulation Libs

import pandas as pd
import itertools
import string
import numpy as np
import seaborn as sns # visualization

# Machine Leanring Libs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Text vectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier # model 1
from sklearn.naive_bayes import MultinomialNB # model 2
from sklearn import metrics # For model evaluation
%matplotlib inline

#Displaying all rows
pd.set_option('display.max_rows',None)

df= pd.read_csv("File-Path")

```
![Screenshot (627)](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/d1c46e76-c54f-4c3a-842a-d189e8055fa9)

### Filtering the Dataset
- For this practice, we will filter the conditions to only about 4 conditions such as Birth Control, Depression, Pain, and Acne and dropping unwanted columns.
```python
#1 Creating a new dataflame
df_train = df[(df['condition']=='Birth Control') | (df['condition']=='Depression') | (df['condition']=='Pain') | (df['condition']=='Acne')]

#2 Create a new dataflame and drop columns we dont need
x=df_train.drop(['uniqueID','drugName','rating','date','usefulCount'], axis=1)

```
![Screenshot (628)](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/169dbf46-3b99-45ff-9ffc-101700ed799f)

### Text cleaning and Preprocessing

- Text Cleaning: Remove noise from the text data, including special characters, punctuation, and irrelevant information.
- Tokenization: Split the text into individual tokens (words or subwords) to facilitate analysis.
- Lemmatization/Stemming: Reduce words to their base or root form to ensure consistency in representation.

The Natural Language Toolkit library, or more commonly NLTK, It supports classification, tokenization, stemming, tagging, parsing, and semantic reasoning functionalities to develop NLP applications and used analyze text data. We need to use it now.

```python
import nltk # .
from nltk .corpus import stopwords 
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

#1 Text cleaning
for i, col in enumerate(x.columns): # Removing tags by replacing
    x.iloc[:, i] = x.iloc[:, i].str.replace('"', '')

#2 Stop words are the sentence construction words such as (is, the, or, then, why, where, )
stop = stopwords.words('english')

```

WordNetLemmatizer
- Purpose: Lemmatization reduces words to their base or dictionary form, known as the lemma. It considers the context and the part of speech (POS) of the word to return the lemma.
- Use Case: Lemmatization is useful when you need the words in their meaningful base form. For example, "running", "ran", and "runs" are all lemmatized to "run"
PorterStemmer
- Purpose: Stemming reduces words to their root form by removing suffixes. It does not consider the context or part of speech and can result in non-dictionary words.
- Use Case: Stemming is useful for scenarios where you need fast and simple normalization, even if the resulting roots are not always meaningful words

```python
# The earlier discused will be put into practice as we continue cleaning and transforming the text.
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

```
The following code function is applied to preprocess a dataset of patient reviews, we will use and apply all variables and libraries wrote ealier in this section. For each review in the dataset, the function will:
- Remove HTML tags to clean the text.
- Remove non-letter characters to simplify the text.
- Convert the text to lowercase for uniformity.
- Remove stopwords to focus on meaningful words.
- Lemmatize the words to their base form for normalization.
- Reassemble the cleaned words into a single string.

```python
# What we are doing is creating a script funtion with certain commands,[We have another another repository for scripts which discuss funtion in details]

from bs4 import BeautifulSoup
import re

def review_to_words(raw_review):
    #1 Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    
    #2 make a space
    letter_only =re.sub('^a-zA-Z', ' ', review_text)
    
    #3 Lower Letters
    words = letter_only.lower().split()
    
    #4 stopwords
    meaningful_words =[w for w in words if not w in stop]
    
    #5 lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    
    #6 Space join words
    
    return(' '.join(lemmitize_words))

# Applying review_to_words function or script that will go threw the reviews and clean the text as commanded inside the function.
x['review_clean']=x['review'].apply(review_to_words)

```

### Splitting dataset for training and testing

```python
features=x['review_clean']
y=x['condition']

X_train, X_test, y_train, y_test = train_test_split(features, y , stratify=y, test_size=0.2, random_state=0)
```
### Confusion matrix plot.
The plot_confusion_matrix function provides a visual representation of the performance of a classification model by plotting the confusion matrix as a heatmap. It can optionally normalize the confusion matrix to show proportions instead of raw counts. This visualization helps in understanding how well the model is performing, where it is making mistakes, and which classes are being confused with each other> 
```python
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color='white' if cm[i, j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

### Feature Engineering 
- Both Bag of Words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency) are techniques used for text representation in Natural Language Processing (NLP). They convert text data into numerical feature vectors that can be used by machine learning algorithms. We will use both to test performance of the models will select

### Model Selection and Training 1: Bag of Words (BoW) with Naive Bayes model
```python
# Using Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Model
mnb = MultinomialNB()
mnb.fit(count_train, y_train)
pred = mnb.predict(count_test)

# Evaluation
score = metrics.accuracy_score(y_test, pred)
print('Accuracy:   %0.3f' % score)

# Plotting Matrix
cm = metrics.confusion_matrix(y_test, pred, labels =['Birth control', 'Depression', 'Pain', 'Acne'])
plot_confusion_matrix(cm, classes=['Birth control', 'Depression', 'Pain', 'Acne'])

```
Accuracy:   0.960
Confusion matrix, without normalization

![download](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/ce2a561d-a1da-4d17-b928-d9f1d218c08d)



### Model Selection and Training 1: Bag of Words (BoW) with PassiveAggressiveClassifier model

```python
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression

# Model
passive = PassiveAggressiveClassifier()
passive.fit(count_train, y_train)
pred = passive.predict(count_test)


# Evaluation
score = metrics.accuracy_score(y_test, pred)
print('Accuracy:   %0.3f' % score)

# Plotting Matrix
cm = metrics.confusion_matrix(y_test, pred, labels =['Birth control', 'Depression', 'Pain', 'Acne'])
plot_confusion_matrix(cm, classes=['Birth control', 'Depression', 'Pain', 'Acne'])
```
Accuracy:   0.970
Confusion matrix, without normalization

![download (1)](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/203ada58-ab27-41e8-a2ee-213c74ec93b4)

METHOD 2: TF-IDF (Term Frequency-Inverse Document Frequency)
```python
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer.transform(X_test)
```

### Model Selection and Training 1: TF-IDF (Term Frequency-Inverse Document Frequency) with Naive Bayes model
```python
# Model
mnb = MultinomialNB()
mnb.fit(tfidf_train_2, y_train)
pred = mnb.predict(tfidf_test_2)


# Evaluation
score = metrics.accuracy_score(y_test, pred)
print('Accuracy:   %0.3f' % score)

# Plotting Matrix
cm = metrics.confusion_matrix(y_test, pred, labels =['Birth control', 'Depression', 'Pain', 'Acne'])
plot_confusion_matrix(cm, classes=['Birth control', 'Depression', 'Pain', 'Acne'])

```
Accuracy:   0.919
Confusion matrix, without normalization
![download (2)](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/20c950ab-5ac8-4658-a9bb-8dbdd59d57d4)


### Model Selection and Training 1: TF-IDF (Term Frequency-Inverse Document Frequency) with PassiveAggressiveClassifier model
```python
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression

# Model
passive = PassiveAggressiveClassifier()
passive.fit(tfidf_train_2, y_train)
pred = passive.predict(tfidf_test_2)


# Evaluation
score = metrics.accuracy_score(y_test, pred)
print('Accuracy:   %0.3f' % score)

# Plotting Matrix
cm = metrics.confusion_matrix(y_test, pred, labels =['Birth control', 'Depression', 'Pain', 'Acne'])
plot_confusion_matrix(cm, classes=['Birth control', 'Depression', 'Pain', 'Acne'])
```
Accuracy:   0.976
Confusion matrix, without normalization
![download (3)](https://github.com/Ackson507/NLP-Natural-Language-Processing-Model/assets/84422970/409d6517-fd0a-48f8-90da-2dbcbdc72264)


### Model Deployment: 
When we have multiple model selected for a project, it is obvious we will move forward with best performing model, in this case we have best model of TF-IDF (Term Frequency-Inverse Document Frequency) with PassiveAggressiveClassifier model with accuracy of 97.6%. After selecting the model then we start modelling it to improve the performance by changing and experimenting with other parameters such bigrames or Trigrames. Deploy the trained model to a production environment using web frameworks or Integrate the deployed model with healthcare systems or electronic health records (EHRs) for seamless adoption by healthcare providers.

### Application

- Disease Prevalence Analysis: Understand the distribution and prevalence of different diseases or medical conditions among patients based on their reviews.
- Early Detection: Identify potential health issues or emerging trends in patient conditions early on, enabling timely intervention and preventive measures.
- Personalized Healthcare: Tailor treatment plans and healthcare services to meet the specific needs of patients based on their reported conditions.




