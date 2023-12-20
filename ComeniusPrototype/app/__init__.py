#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#
import sys
sys.path.insert(0,'/home/jelmer/.local/lib/python3.8/site-packages')
from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
csrf = CSRFProtect(app)
app.secret_key = 'nobodyseeme'

from app import routes
