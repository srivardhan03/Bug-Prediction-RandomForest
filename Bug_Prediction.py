import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate random data
random.seed(100)

def generate_user_feedback():
    feedback_options = [
        "Users reported difficulty logging in.",
        "Search feature occasionally returns no results.",
        "Users experiencing delays in payment processing.",
        "Issues encountered while uploading images.",
        "Users find the dashboard layout confusing.",
        "Some users reported data loss in certain cases.",
        "Users not receiving email notifications.",
        "Difficulties encountered during account creation.",
        "Slow performance reported during peak hours.",
        "Issues with website responsiveness on mobile devices.",
        "Users encountering broken links on the website.",
        "UI freezes reported during navigation.",
        "Some users unable to logout properly.",
        "Issues encountered during checkout process.",
        "Form validation errors reported by users.",
        "Server downtime reported during peak hours.",
        "Difficulties encountered in password reset process.",
        "Broken images appearing on certain pages.",
        "Unclear error messages reported by users.",
        "Sync issues between website and mobile app.",
        "Pages loading slowly reported by users.",
        "Issues encountered while submitting forms.",
        "Users facing problems during subscription renewal.",
        "Problems downloading files from the website.",
        "Compatibility issues with certain web browsers.",
        "Certain functionalities not working as expected.",
        "Users encountering '404 Page Not Found' error.",
        "SSL certificate issues reported by users.",
        "Difficulties encountered during account deletion.",
        "Data synchronization issues between devices."
    ]
    return random.choice(feedback_options)

data = []
for _ in range(1000):
    company_name = random.choice(["TechFusion Inc.", "DataDyne Corp.", "SoftWorks Ltd.", "CyberSolutions", "ByteWise"])
    bug_name = random.choice([
        "SQL Injection Vulnerability",
        "Broken Authentication",
       "Sensitive Data Exposure",
       "XML External Entity (XXE) Injection",
       "Remote Code Execution (RCE)",
       "Cross-Origin Resource Sharing (CORS) Misconfiguration",
       "Clickjacking Vulnerability",
       "Insecure Direct Object Reference (IDOR)",
       "Path Traversal Vulnerability",
       "Denial of Service (DoS) Attack",
       "Server-Side Request Forgery (SSRF)",
       "Content Spoofing Vulnerability",
       "Unrestricted File Upload",
       "Session Fixation Vulnerability",
       "Insufficient Transport Layer Security (TLS)",
       "Security Misconfiguration",
       "Insecure Deserialization",
       "Mass Assignment Vulnerability",
       "Insufficient Authorization",
       "Business Logic Flaw"
    ])
    user_feedback = generate_user_feedback()
    data.append([company_name, bug_name, user_feedback])

df = pd.DataFrame(data, columns=['Company Name', 'Bug Name', 'User Feedback'])

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

X = df.drop(['Bug Name'], axis=1)
y = df['Bug Name']

# Train RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Streamlit app
st.title("Bug Prediction App")

# User input form
company_name_input = st.text_input("Company Name", "DataDyne Corp.")
user_feedback_input = st.text_input("User Feedback", "Users reported difficulty logging in.")
if st.button("Predict"):
    # Predict bug name
    company_name_encoded = label_encoders['Company Name'].transform([company_name_input])[0]
    user_feedback_encoded = label_encoders['User Feedback'].transform([user_feedback_input])[0]
    predicted_bug_encoded = rf_classifier.predict([[company_name_encoded, user_feedback_encoded]])
    predicted_bug_name = label_encoders['Bug Name'].inverse_transform(predicted_bug_encoded)[0]

    # Display prediction
    st.write("Predicted bug:", predicted_bug_name)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    y_pred = rf_classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Bug Name'].classes_, yticklabels=label_encoders['Bug Name'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Display bug frequency
    st.subheader("Bug Frequency")
    bug_frequency = df['Bug Name'].value_counts()
    st.bar_chart(bug_frequency)
