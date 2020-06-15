from project import app,config

if __name__=='__main__':
    app.run(port=5001,debug=bool(int(config['production']['DEBUG'])), host='0.0.0.0', use_reloader=False, threaded=False)
