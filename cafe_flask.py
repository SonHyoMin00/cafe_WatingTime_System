from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import tensorflow.keras as keras
from cafe_model_load import get_xy
import os


# static, templates

app = Flask(__name__)


@app.route('/')
def index():
    res = 0
    return render_template('sample_cafe.html', wait_time=res)


def load_model():
    return keras.models.load_model('model/cafe_multiple_regression_60_0.01.h5')


@app.route('/result', methods=['GET', 'POST'])
def predict_wait_time():
    if request.method == 'POST':
        result = request.form
        li = []
        for _, value in result.items():
            li.append(value)

    model = load_model()
    sca, _, _, data_min, data_max = get_xy()
    li2 = []
    li2.append(int(li[2]))
    li2.append(int(li[3]))
    li2.append(int(li[4]))
    li2.append(int(li[5]))
    li2.append(int(li[6]))
    li2.append(int(li[7]))
    li2.append(int(li[8]))
    li2.append(int(li[9]))

    sum = 0
    for i in range(8):
        sum += li2[i]

    li2.append(sum)
    li2.append(int(li[1]))
    li2.append(int(li[0]))
    li2.append(int(0))
    print(li2)
    v = sca.transform([li2])
    print(v.shape)
    res = model.predict(v[:, :-1])
    res = (data_max - data_min) * res + data_min
    return render_template('sample_cafe.html', wait_time=res[0][0])


if __name__ == '__main__':
    app.run(debug=True)
