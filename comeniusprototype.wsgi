#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,'/var/www/ComeniusPrototype/')
sys.path.insert(1,'/var/www/ComeniusPrototype/ComeniusPrototype/')

from ComeniusPrototype import app as application
application.secret_key = 'nobodyseeme'
