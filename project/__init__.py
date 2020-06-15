import os
from flask import Flask
import logging
from logging import FileHandler,Formatter,getLogger
import configparser

basedir=os.path.abspath(os.path.dirname(__file__))

# initiating the flask app
app=Flask(__name__)

UPLOAD_FOLDER = 'uploads' # all the uploaded files will be saved in this folder which will be in project_color_detection directory

################################################################################

# setting up config function so that we can import configurations from there
def create_config():

    config_file_path = os.path.join(os.path.dirname(__file__), "config/config.ini")

    config=configparser.ConfigParser(interpolation=None)
    config.read_file(open(config_file_path))
    
    return config

# creating config object
config=create_config()

# setting up logger function 
def create_logger(config):
    
    logger=getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = Formatter(config['logging']['format'])
    
    file_handler = FileHandler(config['logging']['logfile'])
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger

###############################################################################

# importing configurations from the config.ini file    
ALLOWED_EXTENSIONS = config['configurations']['ALLOWED_EXTENSIONS']
app.config['UPLOAD_FOLDER']=os.path.join(basedir,'uploads')
IMAGE_SIZE=config['configurations']['IMAGE_SIZE']

       
###############################################################################

# registering the blueprint
from project.routes.routes import api
app.register_blueprint(api)
