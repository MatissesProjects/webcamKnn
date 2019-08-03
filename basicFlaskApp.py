from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET'])
def redirect_to_index():
    return render_template('index_templates.html',
                           numOfClassOne=request.args.get('numOfClassOne', 4),
                           numOfClassTwo=request.args.get('numOfClassTwo', 4),
                           numOfClassThree=request.args.get('numOfClassThree', 4))

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True)
