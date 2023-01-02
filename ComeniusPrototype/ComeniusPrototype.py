from app import app
app.secret_key = 'nobodyseeme'

if __name__ == '__main__':
    import nltk
    app.config['WTF_CSRF_TIME_LIMIT'] = 3600
    app.run(host='0.0.0.0', port=80, use_reloader=False)
