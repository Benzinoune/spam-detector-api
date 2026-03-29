import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from flask import Flask, request, jsonify
import pickle
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

app = Flask(__name__)

# تحميل الملفات التي حفظناها من Colab
model = pickle.load(open('spam_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
config = pickle.load(open('model_config.pkl', 'rb'))

features = config['features']
token_to_index_mapping = config['token_to_index_mapping']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    
    # هنا نضع دالة message_to_count_vector التي شرحناها
    # (يجب تضمين الدالة هنا أيضاً لمعالجة النص القادم من الأندرويد)
    vector = message_to_count_vector(message) 
    vector_scaled = scaler.transform(vector.reshape(1, -1))
    
    prediction = model.predict(vector_scaled)
    result = "Spam" if prediction[0] == 1 else "Ham"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()
