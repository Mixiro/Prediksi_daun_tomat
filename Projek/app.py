from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/bagus.h5')

class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Healthy', 'Septoria Leaf Spot', 'Yellow Leaf Curl Virus'] 

solutions = {
    "Bacterial Spot": "Buang sisa tanaman dan bersihkan alat pertanian secara rutin, rendam benih dalam larutan calcium hypochlorite atau air panas, tanam tanaman non-solanaceae selama ≥1 musim tanam, semprotkan fungisida berbasis tembaga + mancozeb.",
    "Early Blight": "Hancurkan sisa tanaman yang terinfeksi, irigasi tetes: hindari cipratan air ke daun, rotasi tanaman selama 2-3 tahun dengan tanaman bukan famili Solanaceae, fungisida preventif seperti chlorothalonil, azoxystrobin, atau mancozeb.",
    "Healthy": "Penggunaan benih unggul dan tahan penyakit pilih varietas yang memiliki resistensi genetik terhadap TYLCV, Alternaria, dan Phytophthora, rotasi tanaman hindari menanam tomat di lahan yang sebelumnya digunakan untuk tanaman Solanaceae lainnya (cabai, terong) minimal selama 2 musim, pengaturan jarak tanaman hindari kelembaban berlebih dan memungkinkan sirkulasi udara.",
    "Late Blight": "Gunakan kultivar seperti ‘Plum Regal’, ‘Mountain Magic’, fungisida sistemik untuk proteksi dan pengobatan, Sanitasi ketat musnahkan tanaman yang terinfeksi dan lakukan solarisasi lahan, Pengaturan kelembaban ventilasi baik dan penanaman dengan jarak cukup.",
    "Septoria Leaf Spot": "Bersihkan daun sakit dan sisa tanaman, rotasi 1–2 tahun dengan tanaman bukan tomat, jarak tanam dan cagak untuk meningkatkan sirkulasi udara, fungisida chlorothalonil atau mancozeb digunakan secara preventif.",
    "Yellow Leaf Curl Virus": "Gunakan varietas tahan seperti ‘TY20’, ‘Shanty’ atau ‘TYLCV-tolerant lines', gunakan insektisida sistemik, jaring pelindung & mulsa perak untuk mengurangi serangan vektor, sanitasi dan perangkap lengket kuning kontrol populasi vektor dan tanaman inang alternatif."
}

symptoms = {
    "Bacterial Spot": "Bercak air kecil berwarna gelap pada daun yang berkembang menjadi bercak nekrotik, daun menguning, menggulung, dan rontok dini, buah menunjukkan lesi kecil berkerak yang dapat menurunkan kualitas pasar.",
    "Early Blight": "Bercak coklat tua dengan cincin konsentris seperti target pada daun tua, daun menguning dan rontok mulai dari bawah ke atas, dapat menyerang batang dan buah.",
    "Healthy": "Daun berwarna hijau cerah, tidak terdapat bercak, tidak melengkung, dan tidak menggulung, batang kokoh dan tidak berwarna coklat/hitam, Bunga dan buah berkembang normal, tidak rontok premature, tidak ada serangan hama seperti kutu putih, ulat daun, atau trips.",
    "Late Blight": "Bercak air besar berwarna coklat gelap atau kehitaman, bagian bawah daun sering ditumbuhi spora putih di tepinya saat kondisi lembap, infeksi cepat menyebar dan dapat menghancurkan seluruh tanaman.",
    "Septoria Leaf Spot": "Bercak kecil (1–2 mm), bulat, berwarna kelabu dengan tepi gelap, terdapat titik hitam (pycnidia) di tengah bercak, menyebabkan daun menguning dan gugur dari bawah ke atas.",
    "Yellow Leaf Curl Virus": "Daun menggulung ke atas, menguning, dan menebal, tanaman kerdil dan pertumbuhan terhambat, infeksi awal dapat menyebabkan gagal berbuah."
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
        symptom = symptoms.get(label, "Gejala tidak tersedia.")
        solution = solutions.get(label, "Solusi tidak ditemukan.")
        

        return render_template("result.html", label=label, solution=solution, symptom=symptom, image_path=path)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3000)
