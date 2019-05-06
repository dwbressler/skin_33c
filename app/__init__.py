from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'you-will-never-guess'
app.config.from_object(Config) #FIGURE THIS OUT WHEN I DEPLOY FOR REAL

bootstrap = Bootstrap(app)

from app import routes