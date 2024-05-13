# App
This application provides endpoints for ML predictions with the phishing detection model.

## Running the app in a docker container
1. ```docker build . -t flask_service``` 
2. ```docker run -p 8080:8080 flask_service``` 

## Functionality requirements for assignment A2
- Currently hosting the model is not available, for now the model is instead stored in this repository and on the image.
- Flask is used to serve the model and it is runnable in a docker container.
- Currently lib-ml can be imported but does not yet provide the functionality for preprocessing single predictions, therefore its still done on the server.
- Flasgger documentation is provided but does not seem to work from within a docker container right now.