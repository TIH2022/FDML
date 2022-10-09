# Fraud Detection using AWS SageMaker (Machine Learning)

Nowadays with technology businesses moving online, fraud and abuse in online systems is constantly increasing as well. Traditionally, rule-based fraud detection systems are used to combat online fraud, but these rely on a static set of rules created by Subject experts. This FDML project uses machine learning(SageMaker) to create models for fraud detection that are dynamic, self-improving and maintainable. Importantly, they can scale with the online business.

Specifically,here we are going to show how to use Amazon SageMaker to train supervised and unsupervised machine learning models on historical transactions, so that they can predict the likelihood of incoming transactions being fraudulent or not. We also show how to deploy the models, once trained, to a REST API that can be integrated into an existing business software infrastructure. This FDML project includes a demonstration of this process using a public, anonymized credit card transactions [dataset provided by ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud), but can be easily modified to work with custom labelled or unlaballed data provided as a relational table in csv format.


## Architecture

The project architecture deployed by the cloud formation template is shown here.

![](deployment/architecture.png)

## Project Description
The FDML project uses Amazon SageMaker to train both a supervised and an unsupervised machine learning models, which are then deployed using Amazon Sagemaker-managed endpoints.

If you have labels for your data, for example if some of the transactions have been annotated as fraudulent and some as legitimate, then you can train a supervised learning model to learn to discern the two classes. In this project, we provide a recipe to train a gradient boosted decision tree model using [XGBoost on Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html). The supervised model training process also handles the common issue of working with highly imbalanced data in fraud detection problems. The project addresses this issue into two ways by 1) implementing data upsampling using the "imbalanced-learn" package, and 2) using scale position weight to control the balance of positive and negative weights.

If you don't have labelled data or if you want to augment your supervised model predictions with an anomaly score from an unsupervised model, then the project also trains a [RandomCutForest](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html) model using Amazon SageMaker. The RandomCutForest algorithm is trained on the entire dataset, without labels, and takes advantage of the highly imbalanced nature of fraud datasets, to predict higher anomaly scores for the fraudulent transactions in the dataset.

Both of the trained models are deployed to Amazon SageMaker managed real-time endpoints that host the models and can be invoked to provide model predictions for new transactions.

The model training and endpoint deployment is orchestrated by running a [jupyter notebook](source/notebooks/sagemaker_fraud_detection.ipynb) on a SageMaker Notebook instance. The jupyter notebook runs a demonstration of the project using the aforementioned anonymized credit card dataset that is automatically downloaded to the Amazon S3 Bucket created when you launch the solution. However, the notebook can be modified to run the project on a custom dataset in S3. The notebook instance also contains some example code that shows how to invoke the REST API for inference.

In order to encapsulate the project as a stand-alone microservice, Amazon API Gateway is used to provide a REST API, that is backed by an AWS Lambda function. The Lambda function runs the code necessary to preprocess incoming transactions, invoke sagemaker endpoints, merge results from both endpoints if necessary, store the model inputs and model predictions in S3 via Kinesis Firehose, and provide a response to the client.


## Contents

* `deployment/`
  * `fraud-detection-using-machine-learning.yaml`: Creates AWS CloudFormation Stack for solution
* `source/`
  * `lambda`
    * `model-invocation/`
      * `index.py`: Lambda function script for invoking SageMaker endpoints for inference
  * `notebooks/`
    * `src`
      * `package`
        * `config.py`: Read in the environment variables set during the Amazon CloudFormation stack creation
        * `generate_endpoint_traffic.py`: Custom script to show how to send transaction traffic to REST API for inference
        * `util.py`: Helper function and utilities
    * `sagemaker_fraud_detection.ipynb`: Orchestrates the solution. Trains the models and deploys the trained model
    * `endpoint_demo.ipynb`: A small notebook that demonstrates how one can use the solution's endpoint to make prediction.
  * `scripts/`
    * `set_kernelspec.py`: Used to update the kernelspec name at deployment.
  * `test/`
    * Files that are used to automatically test the solution


1. What problem are you addressing with your project?(Development Field)
2. Please describe the technologies being used in your project.(Development Field)
3. Round 1 - Submit your Elevator Pitch Attachments(Development Field)
4. Round 1 - Submit your Elevator Pitch Links or Additional Information(Development Field)
5. Round 1 - Additional Attachments(Development Field)
6. I acknowledge that I have read and understood the Participant Terms *Yes
7. I acknowledge that I have submitted Round 1 files as per the TIH2022 guidelines.(Development Field)


