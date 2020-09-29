from app import app
app.secret_key = 'you-will-never-guess'

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
