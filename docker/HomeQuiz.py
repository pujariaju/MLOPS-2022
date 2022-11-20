from flask import Flask
from flask import request
from joblib import load
import os

app = Flask(__name__)
model_path = "SVC(C=10, gamma=0.0001)"
print(model_path)
model = load(model_path)

@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    print("done loading")
    predicted1 = model.predict([image1])
    
    image2 = request.json['image2']
    print("done loading")
    predicted2 = model.predict([image2])

    if(predicted1[0]==predicted2[0]):
        return "Same Digits"
    else :
        return "Not Same"
    #return {"y_predicted":int(predicted1[0])}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)