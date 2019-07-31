from flask import Flask, render_template
app = Flask(__name__)


@app.route('/', methods=['GET'])
def redirect_to_index():
    return render_template('index_templates.html', message='asdf')

@app.route('/basic_calc/<a>/<b>')
def do_calc(a,b):
    solution = float(a) + float(b)
    return "{} + {} = {}".format(a,b,solution)

if __name__ == '__main__':
    app.run()
