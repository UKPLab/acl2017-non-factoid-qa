# Candidate Ranking

This application contains a framework to integrate, train and load answer selection approaches as described in our
related demonstration paper. 

 
## Modules 
The framework contains four different types of components that are pulled together at run-time.
 
#### Data
Data modules read the dataset that contains examples to train and evaluate the answer selection approach. Each data 
module inherits from the _experiment.Data_ class. This project ships with several pre-built data modules that can read
the following datasets:

   - InsuranceQA Version 1
   - InsuranceQA Version 2
   - A special TSV format. The project _Candiate Retrieval_ can be used to create a dataset for any StackExchange 
     archive with this format.

#### Model
Model modules contain the logic to build the tensorflow graphs that learn the representations of questions and candidate
answer given an input of word embedding indices. Models usually inherit from ```experiment.qa.model.QAModel```, but at 
least from ```experiment.Model```. We bundle the application with four different models:
 
   - __Embeddings:__ The most simple model, which learns the representation by averaging over all word embeddings
   - __BiLSTM:__ An LSTM model with 1-max pooling to learn the representation
   - __APBiLSTM:__ A model that uses Attentive Pooling with LSTM (Dos Santos et al., 2016).
   - __LW-BiLSTM:__ Uses separate BiLSTMs to determine the importance of segments in the input (Rücklé & Gurevych, 2017) 
   - __WordImportanceBiLSTM:__ A simple extension of BiLSTM that learns a weight for each input word. This allows it to
     ignore stopwords and put a higher emphasis on more important words.

Only the last two models contain attention mechanisms (but rather different ones). 

It is very easy to add new models with attention mechanisms to this framework. To work with our visualization, it is
required to assign a weight to each position of the input. These weights must be provided in a float tensor of shape 
[batch, text-length]. They must be assigned to specific parameters in the model class:

```python
    self.question_importance_weight = my_tensor 
    self.answer_good_importance_weight = my_other_tensor 
```

See _APBiLSTM_ or _WordImportanceBiLSTM_ for an example.

#### Training 
Models are trained with the examples read by the data module. We bundle several training modules:
   
   - __QANoTraining:__ Can be used to skip training (e.g. when using the average embeddings model with pretrained 
     embeddings)
   - __QATrainingSimple:__ A simple, batched training process
   - __InsuranceQATrainingDynamic:__ A dynamic training approach as described by Tan et al. (2016)


#### Evaluation
We offer only one evaluation module. It measures Accuracy, MRR, and MAP.

## Configuration
We use YAML files to describe all necessary parameters to run an experiment. This allows researchers to fully configure 
and combine the previously described modules without ever having to change a single line of code. 

An example configuration file with detailed documentation is available in the _config_ folder.

__Important__: We use a fallback config file named _default_config.yaml_. Do not remove or change this file.
  

## Setup

   - Install Tensorflow 1.2 (or similar)
   - Run ```pip install -r requirements.txt``` to install all other dependencies 
   - We use NLTK for tokenization. You may need to download the package "punkt"
     - `python -m nltk.downloader punkt`
   - Download word embeddings from http://nlp.stanford.edu/data/glove.6B.zip
   
## Running the Application

   - First, create a YAML configuration file that describes your desired setup (dataset, model, hyperparameters)
   - Train the model running ```python run_experiment.py <path/to/config.yaml>```
   - You can then start the webserver that allows the web-frontend to re-rank candidates with your pre-trained model
       ```python run_reranking_server.py <path/to/config.yaml> --port=5001```
       - Alternatively, you could also skip the model training (second step) and directly run the reranking server. It 
       will detect that the model was not trained and will start the training process first. 


## References

   - Andreas Rücklé and Iryna Gurevych. 2017. Representation Learning for Answer Selection with LSTM-Based 
   Importance Weighting. In Proceedings of the 12th International Conference on Computational Semantics (IWCS 2017).
   (to appear).
   - Cicero Dos Santos, Ming Tan, Bing Xiang, and Bowen Zhou. 2016. Attentive Pooling Networks. arXiv preprint 
   https://arxiv.org/abs/1602.03609.
   - Ming Tan, Cicero Dos Santos, Bing Xiang, and Bowen Zhou. 2016. Improved representation learning for question answer
   matching. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL). 
   pages 464–473.
