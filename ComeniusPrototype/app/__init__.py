import sys
sys.path.insert(0,'/home/jelmer/.local/lib/python3.8/site-packages')
from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
csrf = CSRFProtect(app)
app.secret_key = 'nobodyseeme'

from app import routes
