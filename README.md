# Big-Data

In this project, we designed and implemented a Real-Time Fake News Detection System that combines the power of big data processing using PySpark with machine learning techniques to effectively identify and classify fake news articles. The goal was to build a scalable, accurate, and real-time system capable of detecting misinformation as it is published online.

We began by selecting the WEL Fake dataset, a robust and diverse collection of over 72,000 news articles, aggregated from four popular datasets: Kaggle Fake News Dataset, McIntire Fake News Dataset, Reuters News Dataset, and BuzzFeed Political News Dataset. This comprehensive dataset includes both real and fake news articles, making it ideal for training models with strong generalization ability.

The first step in our pipeline was data preprocessing, which was essential for cleaning and standardizing the raw text data. We removed punctuation, special characters, and stop-words, and converted all text to lowercase. Then, we tokenized the text and applied lemmatization to reduce words to their root forms. The cleaned text was transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which helps the model identify the most important terms in each article.

We evaluated several machine learning algorithms, including Logistic Regression, Decision Tree Classifier, and Random Forest Classifier. After splitting the dataset into 80% training and 20% testing data, we trained each model and assessed them using standard evaluation metrics: accuracy, precision, recall, and F1-score. Our experiments showed that Logistic Regression achieved the best results, with an accuracy of 97.22%, precision of 97.31%, and a recall and F1-score of 97.22%. Therefore, this model was selected for real-time deployment.

We then moved on to building the real-time streaming pipeline using PySpark Streaming. This component enables continuous ingestion and classification of incoming news articles. The real-time data undergoes the same preprocessing and TF-IDF transformation steps as the training data. The pre-trained Logistic Regression model is then used to predict whether the article is real or fake in real time.

The classification results are stored in MongoDB, a NoSQL database suitable for handling unstructured data efficiently. Each entry in the database contains the news text, its predicted label (real or fake), and a timestamp to track trends over time. Additionally, predictions are exported to JSON files to facilitate further analysis and visualization.

To provide users with an accessible and interactive interface, we developed a Streamlit-based web application. This application includes:

Live fake news detection, allowing users to input custom news content and receive instant predictions.

Visual analytics, including confusion matrices, bar charts of accuracy and other performance metrics, and tables displaying stored predictions.

A summary view of the top 5 most recent real and fake news articles.

An easy-to-use dashboard for exploring trends in misinformation.

By integrating big data tools, machine learning, and a user-friendly web interface, this project presents a powerful solution to the growing problem of fake news. Our system is capable of real-time classification, is scalable for large datasets, and provides valuable insights through interactive visualization. This makes it suitable for use in media organizations, social platforms, and public fact-checking services.
