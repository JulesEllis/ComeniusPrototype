from app import app
app.secret_key = 'nobodyseeme'

if __name__ == '__main__':
    import nltk
    nltk.download('punkt')
    app.run(host='0.0.0.0', port=80)
