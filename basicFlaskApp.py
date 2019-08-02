from flask import Flask, render_template
app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/<numOfClassOne>', methods=['GET'])
@app.route('/<numOfClassOne>/<numOfClassTwo>', methods=['GET'])
def redirect_to_index(numOfClassOne=4, numOfClassTwo=4):
    return render_template('index_templates.html', numOfClassOne=numOfClassOne, numOfClassTwo=numOfClassTwo)

@app.route('/basic_calc/<a>/<b>')
def do_calc(a,b):
    solution = float(a) + float(b)
    return "{} + {} = {}".format(a,b,solution)

if __name__ == '__main__':
    app.run()
