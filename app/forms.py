from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

from flask import url_for, redirect, render_template
from flask_wtf import Form
from flask_wtf.file import FileField
from werkzeug import secure_filename

#from flask_wtf.file import FileField, FileRequired
#from werkzeug.utils import secure_filename

class QueryForm(FlaskForm):
    #the_document = StringField('Document', validators=[DataRequired()])
    the_wik_search = StringField('Search for a Wikipedia Page (e.g. "Janis Joplin")', default="Janis Joplin", validators=[DataRequired()])
    the_query = StringField('Your Question (e.g. "When was she born")', default="When was she born?", validators=[DataRequired()])
    submit = SubmitField('Submit')

#class PhotoForm(FlaskForm):
#    photo = FileField(validators=[FileRequired()])

class UploadForm(Form):
    file = FileField()