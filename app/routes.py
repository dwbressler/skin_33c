from flask import render_template, flash, redirect
from app import app
from app.forms import QueryForm

#from app.pretrainedBERT.examples import run_squad
import sys
sys.path.append('./fastai')

from fastai import *
from fastai.vision import *
from fastai.tabular import *
from fastai.vision.learner import cnn_config

import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import torchvision.transforms as transforms

import time
import importlib
import warnings
import os, re

#from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

this_is_cpu=0
DATAPATH = Path('/Users/davidbressler/pythonstuff/local_FD/')

#from bs4 import BeautifulSoup

#CHANGES: This step produces warning "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."

#CHANGES: WARNING: Do not use the development server in a production environment. Use a production WSGI server instead.

#import pytorch_pretrained_bert.examples.run_squad

#import numpy as np
#import torch
#import os
#import collections

#set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model (weights)
# filename=os.path.join(app.root_path, 'models', 'pytorch_model.bin')
# model_state_dict = torch.load(filename, map_location='cpu')
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',state_dict=model_state_dict)
# model.to(device) 
# model.eval()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    print(app.root_path)
    form= QueryForm()
    if form.validate_on_submit():
        
        the_answer='bla'
        wik_url="https://en.wikipedia.org/wiki/Janis_Joplin"

        return render_template('index.html',title='Home', form=form, wik_url=wik_url, the_wik_search=None, the_query=None, the_answer=the_answer)

        #flash('Your Query: {}'.format(
        #    form.the_query.data))
        #flash('The Document: {}'.format(
        #    form.the_document.data))
        #return redirect('/index')
    return render_template('index.html',title='Home', form=form, wik_url="https://en.wikipedia.org/wiki/Janis_Joplin", the_wik_search=None, the_query=None, the_answer="January 19, 1943")


