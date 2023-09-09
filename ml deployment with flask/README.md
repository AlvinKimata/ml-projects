## This project involves building and deploying a machine learning application with flask as an API endpoint.
## Reference to this project can be found [here.](https://www.datacamp.com/tutorial/machine-learning-models-api-python)

### The command below uses cURL command to perform a POST request to the endpoint to perform inference.
```bash
curl -X POST -H "Content-Type: application/json" -d '{"Age": 30, "Sex": "male", "Embarked": "C"}' http://localhost:12345/predict
```

### The output or result of the command below is returned as a json object: 

```text
{
  "prediction": "[0]"
}

```