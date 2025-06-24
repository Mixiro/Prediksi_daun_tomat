from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/bagus.h5')

class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Healthy','Septoria Leaf Spot','Yellow Leaf Curl Virus'] 
solutions = {
    "Bacterial Spot": "Gunakan fungisida berbasis tembaga dan hindari penyiraman dari atas untuk mengurangi kelembaban.",
    "Early Blight": "Buang daun yang terinfeksi, rotasi tanaman, dan gunakan fungisida berbahan aktif klorotalonil.",
    "Healthy": "Tanaman sehat, tetap lakukan pemantauan rutin dan pemupukan sesuai jadwal.",
    "Late Blight": "Cabut dan musnahkan tanaman yang terinfeksi, gunakan fungisida sistemik seperti mankozeb.",
    "Septoria Leaf Spot": "Buang daun bawah yang terkena, hindari penyiraman berlebih, dan gunakan fungisida.",
    "Yellow Leaf Curl Virus": "Gunakan insektisida untuk mengendalikan vektor (kutu putih), dan tanam varietas tahan virus."
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        path = os.path.join("static/uploaded", file.filename)
        file.save(path)

        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        label = class_names[class_id]
        solution = solutions.get(label, "Solusi tidak ditemukan.")

        return render_template("result.html", label=label, solution=solution, image_path=path)
        pass
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3000)
