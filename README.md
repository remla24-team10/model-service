# 🔧 model-service
This application provides endpoints for ML predictions with the phishing detection model.

## 🐋 Running the app in a Docker container
1. ```docker build . -t flask_service``` 
2. ```docker run -p 8080:8080 flask_service``` 

## 📝 Functionality requirements for assignment A2
- Model tokenizer and encoder is hosted on google drive and downloaded during runtime.
- Flask is used to serve the model and it is runnable in a docker container.
- Utilises pre and postprocessing functions provided by lib-ml.
- Flasgger documentation is provided but does not seem to work from within a docker container right now.