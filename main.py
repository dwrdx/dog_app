import os
from bottle import Bottle, template, static_file, request, redirect
from predictor import funny_dog_breed_predictor

app = Bottle()

save_path = './static'


def funny_predictor(img_path):
    is_human, is_dog, breed = funny_dog_breed_predictor(img_path)
    dogname = breed 
    return is_dog, dogname

@app.get('/static/<filename>')
def server_static(filename):
    return static_file(filename, root='./static')

@app.get('/')
def index():
    return template('index.tpl', filename='Husky.jpeg', human_or_dog='dog', dogname='Husky')

@app.get('/<filename>')
def index(filename):
    is_dog, dogname = funny_predictor(save_path + '/' + filename)
    if is_dog:
        human_or_dog = 'dog'
    else:
        human_or_dog = 'human'
    return template('index.tpl', filename=filename, human_or_dog=human_or_dog, dogname=dogname)

@app.post('/upload')
def upload():
    upload     = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    
    upload.save(save_path, overwrite=True) # appends upload.filename automatically

    redirect("/"+upload.filename)



if __name__ == "__main__":
    port=int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)