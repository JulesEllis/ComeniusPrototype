#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#
from app import app
app.secret_key = 'nobodyseeme'

if __name__ == '__main__':
    import nltk
    app.config['WTF_CSRF_TIME_LIMIT'] = 3600
    app.run(host='0.0.0.0', port=80, use_reloader=False)
