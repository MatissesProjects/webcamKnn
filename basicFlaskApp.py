from flask import Flask, render_template
app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/<int:numOfClassOne>', methods=['GET'])
@app.route('/<int:numOfClassOne>/<int:numOfClassTwo>', methods=['GET'])
def redirect_to_index(numOfClassOne=4, numOfClassTwo=4, numOfClassThree=4):
    return render_template('index_templates.html',
                           numOfClassOne=numOfClassOne, numOfClassTwo=numOfClassTwo, numOfClassThree=numOfClassThree)

if __name__ == '__main__':
    app.run()
