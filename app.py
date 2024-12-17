from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Example complexity matrix data
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    return render_template('index.html', matrix=matrix)

if __name__ == '__main__':
    app.run(debug=True)
