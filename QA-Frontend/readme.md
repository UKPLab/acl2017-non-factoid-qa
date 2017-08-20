# QA Frontend

This application coordinates the two other applications Candidate Retrieval and Candidate Ranking. It also bundles the 
web user interface.


## Requirements

   * Python 3
   * [NodeJS](https://nodejs.org)

 
## Setup

The QA Frontend uses modern web techniques such as typescript, angular2, sass. It relies on several 3rd party libraries 
and  frameworks. Before starting the application, those libraries and dependencies need to be installed.

  * Run ```pip install requirements.txt``` to install all required python packages
  * Install [NodeJS](https://nodejs.org)
  * cd to the folder _static_ fetch all frontend dependencies using ```npm install```
    * This should automatically compile all required .js and .css files
    * However, if no output files were generated you might need to install typescript globally: ```npm install typescript -g```
    * You can also re-run the compile commands individually: ```npm run tsc``` and ```npm run sass```


## Running the Application

  * Before starting the frontend application, adapt config.yaml to your needs. See the config file for further information
  * You can now start the webapplication by running ```python main.py``` from the root folder of QA Frontend. The other 
    components (Candidate Retrieval and Candidate Ranking) must be started before entering any queries.
     * To enable the comparison of neural attention weights of different models, start multiple candidate ranking application
       instances and register each one in the YAML configuration file of the frontend.
 