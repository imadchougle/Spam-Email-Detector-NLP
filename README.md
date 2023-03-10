# Spam-Email-Detector-NLP
 Spam Email detection using Natural Language Processing


Abstract

Email Spam has become a major problem nowadays, with the Rapid growth of internet
users, Email spam is also increasing. People are using them for illegal and unethical
conduct, phishing, and fraud. Sending malicious links through spam emails can harm our
system and can also seek into your system. So, it is needed to Identify those spam mails
which are fraudulent, this project will identify those spam by using a naive base
classifier, we have implemented and applied this algorithm to our data sets and this
algorithm is selected for email spam detection.

Introduction

We get hundreds of messages from unknown sources and our inbox is filled with
unwanted emails. These unwanted messages are called spam and essential messages
are called ham mail. We will prepare a model that will categorize messages on the
website as spam or ham. To achieve this, data from the messages is to be collected first
and natural language processing techniques are to be applied to it. Spam filtering
among messages helps the user to have a good visualization of the message.
Unnecessary messages will be marked as spam so users need not waste their time
reading them. In this project, we propose to classify data in the messages as either spam
(unwanted) or ham(wanted) messages. We devised our spam detector.

![0_SHZ7ehjxCaS7-VBp](https://user-images.githubusercontent.com/54437743/224354937-45de1671-d9d4-4f34-8592-7a625ce25eed.png)


Figure 1 : Email spam Detection diagram

Objective
The objective of email spam detection using NLP is to develop a model that can
accurately classify incoming emails as either spam or not spam (also known as ham)
based on the textual content of the email. This involves using natural language
processing (NLP) techniques to preprocess the text data, extract relevant features, and
train a machine learning algorithm to classify the emails. The ultimate goal is to reduce
the number of unwanted and potentially harmful emails that make it into users&#39;
inboxes, thereby improving email security and efficiency.

Dataset

Let’s start with our spam detection data. We’ll be using the open-source Spam base
dataset from the UCI machine learning repository, a dataset that contains 5569 emails,
of which 745 are spam.
The target variable for this dataset is ‘spam’ in which a spam email is mapped to 1 and
anything else is mapped to 0. The target variable can be thought of as what you are
trying to predict. In machine learning problems, the value of this variable will be
modeled and predicted by other variables.

Naive Bayes classifier
 The Naive Bayes algorithm is a supervised learning algorithm, which is based on
the Bayes theorem and used for solving classification problems.
 It is mainly used in text classification that includes a high-dimensional training
dataset.
 Naive Bayes Classifier is one of the simple and most effective Classification
algorithms which helps in building fast machine learning models that can make
quick predictions.
 It is a probabilistic classifier, which means it predicts based on the probability of
an object.
 Some popular examples of the Naive Bayes Algorithm are spam filtration,
Sentimental analysis, and classifying articles.

Problem
 Time and productivity loss: Spam emails can be time-consuming and distracting,
causing users to spend more time sorting through their emails reducing overall
productivity.
 Security risks: Spam emails can contain viruses, malware, or links to phishing
sites, which can lead to identity theft, data breaches, and other security risks.
 Email server overload: large volumes of spam emails can overload email servers,
reducing their efficiency and potentially causing them to crash.

Workflow of project

![download](https://user-images.githubusercontent.com/54437743/224355332-0da6abf3-ae35-43fc-9732-f48c4582f951.png)

Figure 2 : Flowchart

1. Data Preprocessing: The first step is to preprocess the email text data by
removing stop words, punctuation, and other non-relevant text elements. This
step also includes tokenization, stemming, and lemmatization to normalize the
text data.
2. Feature Extraction: Next, relevant features are extracted from the preprocessed
text data. This can include bag-of-words representations, TF-IDF vectors, and n-
grams.
3. Split Data: The dataset is split into training and testing sets, typically in a 70:30
ratio. The training set is used to train the Naive Bayes model, while the testing set
is used to evaluate the model&#39;s performance.
4. Train Model: The Naive Bayes algorithm is used to train the model on the training
set. This involves calculating the probabilities of each word in the email being
spam or not spam, based on the frequency of occurrence of those words in the
training set
5. Prediction: Once the model has been evaluated and validated, it can be used to
predict whether a new incoming email is a spam or not.

6. Deployment: Once the model has been optimized, it can be deployed in a
production environment to classify incoming emails in real time.
Implementation

![Screenshot 2023-03-10 152529](https://user-images.githubusercontent.com/54437743/224355662-d8f9044b-7106-45b1-bcd0-9cd2bc43043f.jpg)

Figure 3 : Project Implementation (Entered data is classified as spam by the software)

![Screenshot 2023-03-10 152610](https://user-images.githubusercontent.com/54437743/224355799-523c5fa5-928d-4aba-8fd7-c05765ab806c.jpg)

Figure 4 : Project Implementation (Entered data is classified as not a spam by the software)

In our project we have made a flask website which takes input from the user and
classifies if the email is spam or not a spam using naive bayes algorithm.
Conclusion
In conclusion, email spam detection using NLP techniques such as CountVectorizer and
Naive Bayes is an important application of machine learning and natural language
processing. By automatically identifying and filtering out spam emails, this technology
helps improve the efficiency and security of email communication. As can see, datasets
that have fewer instances of e-mails and attributes can give good performance for the
Naive Bayes classifier. You can see that the results obtained are quite good.



References
G. Revathi, K. N. (2022). Email Spam Detection using Naïve Bayes Algorithm. International Journal for
Research in Applied Science &amp; Engineering Technology (IJRASET).
https://en.wikipedia.org/wiki/Naive_Bayes_classifier
