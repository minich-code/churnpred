from flask import Flask, request, jsonify, render_template
import pandas as pd 
from flask_mail import Mail, Message 
from pymongo import MongoClient 
import os 
from src.churn.pipelines.pip_07_prediction_pipeline import CustomData, PredictionPipeline  # Ensure correct import paths
from src.churn.exception import FileOperationError
from src.churn import logging

# Creating the Flask app
app = Flask(__name__)

# Configure mailtrap 
app.config['MAIL_SERVER']='bulk.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = '6d0d5e3485f30cbe048871e1b9c7c2c9'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# Mongo connection string 
MONGO_URI = "mongodb+srv://Minich:Mydatabase.456@minich-data-repository.gzlkk1s.mongodb.net/"

# Route for homepage 
@app.route('/')
def home():
    return render_template('home.html')

# Route for uploading data and making the prediction

@app.route('/predict', methods=['GET', 'POST'])
def predict_data_point():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        try:
            if 'csv_file' not in request.files:
                logging.error("No csv_file key in request.files")
                return jsonify({"error": "No file part in the request"}), 400
            
            csv_file = request.files['csv_file']

            if csv_file.filename == '':
                logging.error("No selected file")
                return jsonify({"error": "No file selected"}), 400
            
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            if 'name' not in df.columns or 'email' not in df.columns:
                logging.error("Required columns 'name' or 'email' not found in the CSV file")
                return jsonify({"error": "Required columns not found"}), 400 
            
            names = df['name']
            emails = df['email']

            df = df.drop(columns=['name','email'])

            # Initialize the prediction pipeline 
            prediction_pipeline = PredictionPipeline()

            # Make predictions 
            predictions = prediction_pipeline.make_predictions(df)

            # Add the predictions to the dataframe 
            df['churn_risk_score'] = predictions

            # Re-attach name and email 
            df['name'] = names
            df['email'] = emails

            # Update the database 
            update_database(df)

            # Send email notifications to the recipients 
            for i, row in df.iterrows():
                send_email(row['email'], row['name'], row['churn_risk_score'])


            return render_template('results.html')

        except FileOperationError as e:
            logging.error(f"Error occurred while reading file: {str(e)}")
            return jsonify({"error": str(e)}), 400
        

def send_email(recipient, name, prediction):
    subject = "Account Status Notification"

    if prediction == 2:
        text = f"Dear {name}, we are happy to inform you that your subscription is in good standing."

    elif prediction == 1:
        text = f"Dear {name}, your subscription is nearing renewal. Please take the necessary action. Contact support for additional help."

    else:
        text = f"Dear {name}, Your subscription has expired. Please renew your subscription to continue enjoying our services."

    
    msg = Message(subject, sender='mailtrap@demomailtrap.com', recipients=[recipient])
    msg.body = text

    with app.app_context():
        mail.send(msg)


def update_database(df):
    """
    Saves the transformed DataFrame to a new MongoDB collection
    """
    client = MongoClient(MONGO_URI)
    db = client['churn']
    collection = db['production_churn_data'] # New collection name 
    
    # Convert the dataframe to a dictionary and insert to MongoDB
    data_dict = df.to_dict("records")
    collection.insert_many(data_dict)
    print("Data saved to MongoDB collection 'production_churn_data'") 


# Run the Flask app 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)






