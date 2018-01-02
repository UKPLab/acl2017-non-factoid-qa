# Candidate Retrieval

This application allows to index all answers from a dataset (StackeExchange or InsuranceQA) with Elastic Search. It 
provides a webservice that allows to retrieve similar answers for a query-question.

Additionally, it also provides a functionality to create new datasets to train the candidate ranking application using
StackExchange archives.


## Requirements

   * Scala 2.11
   * Elasticsearch


## Setup

You can start the candidate ranking service on your local machine with ```sbt play:run```.

If you want to deploy it to a different machine, you first need to build and package the application:

   * Compile the application with ```sbt play:dist``` 
   * A zip archive containing the compiled assets is now available in _target/universal_
   * Unzip the archive on your target machine and edit the configuration in _conf/application.conf_. A description of 
     the relevant options is available in the provided configuration file
   * Start elasticsearch
   * run ```bin/candidate-retrieval``` (within the unzipped archive) to start the application
      * For some platforms it might be required to run ```bin/candidate-retrieval -J-Dfile.encoding=UTF8``` instead. See [Issue #5](https://github.com/UKPLab/acl2017-non-factoid-qa/issues/5)


## Usage

   * Once you have configured a dataset in the configuration file _conf/application.conf_, you can index all alswers by 
      navigating to the URL ```/create-index``` (e.g. ```http://localhost:9000/create-index```). Make sure that this 
      route is not publicly accessible.
      * Info: When indexing InsuranceQA V2, the application might require more than 1GB memory.
   * When the dataset is indexed, the URL ```/query?q=test&n=100``` allows to retrieve candidate answers.
   * You can easily extend the application with new data readers to support other data sources. See _app.data.readers_.

   
### Dataset Creation

We have included a method to create new datasets based on StackExchange to train the candidate ranking application. 
Based on the currently indexed data (answers) as well as the configured data source, you can create new datasets for
training using the route ```/write-tsv-dataset```. For more information see the configuration file.

Note: Dataset creation may take a while since we are querying elasticsearch for each individual question in the data
source. Take a look at the logfile to see the progress.
