from flask import Flask, request, jsonify
import pickle
import numpy as np
import nltk
import os

# تحميل الأدوات الضرورية لـ NLTK
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r"\w+")
except Exception as e:
    print(f"NLTK Download Error: {e}")

app = Flask(__name__)

# دالة لتحميل الملفات بأمان
def load_pickle(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Missing file: {file_name}")
    return pickle.load(open(file_name, 'rb'))

try:
    model = load_pickle('spam_model.pkl')
    scaler = load_pickle('scaler.pkl')
    config = load_pickle('model_config.pkl')
    features = config['features']
    token_to_index_mapping = config['token_to_index_mapping']
    stopwords = config['stopwords']
    print("All models loaded successfully!")
except Exception as e:
    print(f"Model Loading Error: {e}")

def message_to_token_list(s):
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t not in stopwords]
    return useful_tokens

def message_to_count_vector(message):
    count_vector = np.zeros(len(features))
    processed_list_of_tokens = message_to_token_list(message)
    for token in processed_list_of_tokens:
        if token in features:
            index = token_to_index_mapping[token]
            count_vector[index] += 1
    return count_vector

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التأكد من وصول البيانات
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        print(f"Received message: {message}") # سيظهر في Logs

        # تحويل ومعالجة
        vector = message_to_count_vector(message)
        vector_scaled = scaler.transform(vector.reshape(1, -1))
        
        # التنبؤ
        prediction = model.predict(vector_scaled)
        result = "Spam" if prediction[0] == 1 else "Ham"
        
        return jsonify({'prediction': result})

    except Exception as e:
        # هذا السطر سيكشف لنا الحقيقة في Render Logs
        print(f"CRITICAL ERROR: {str(e)}") 
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
