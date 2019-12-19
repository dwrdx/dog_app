# dog_app - funny dog breed predictor

This simple web application is built with bottlepy framework and a trained CNN predictor. The application allows the user to upload an image which may contains either a dog or a human in It, then the application can tell the user the breed of the actual dog in the picture or of the dog that the human looks like.

# How it works
* The pre-trained CNN parameters are saved in weights.best.VGG19.hdf5 file, the application only loads the parameters to get back the model.
* opencv is included to detect fi there is a human in the image
* ResNet50 is included to detect if there is a dog in the iamge
* our model is transferred from VGG19 with special trainings to identify the breed of the dog

# How to run the project
1. Clone the project
2. Created a python virtual env(optional)
3. pip install -r requirements.txt
4. python main.py

# Deploy
Procfile is included. So the project can be easily deployed to Heroku with a single click on "Deploy" on the Heroku deploy page. A sample deplyment can be found here [funny-dog-breed-predictor](https://funny-dog-breed-predictor.herokuapp.com/)