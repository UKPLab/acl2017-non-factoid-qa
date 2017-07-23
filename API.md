# Candidate-Retrieval

Public API


## Query Candidate Answers

Returns json data for a number of retrieved candidate answers.

  * __URL:__ 

    /query

  * __Method:__ 

    GET

  * __URL Params:__

    __Required:__

      `q=[string]` the query

    __Optional:__

      `n=[int]` number of candidates to retrieve (default=500)

  * __Success Response:__

    * __Code:__ 200

      __Content:__ 

      ```
      {
        "candidates": [
          "candidate answer 1 text", 
          "candidate answer 2 text", 
          ...
        ]
      }
      ```


# Answer Ranking

Public API


## Re-Rank Candidate Answers

Re-ranks a list of candidate answers according to a question and returns json data for a number of the re-ranked candidate answers together with all the associated attention weights.

  * __URL:__ 

    /rank

  * __Method:__ 

    POST

  * __URL Params:__

    __Optional:__

      `n=[int]` number of re-ranked candidate answers to return (default=10)

  * __Data Params:__

    We expect JSON payload of the following format:

    ```
    {
      "question": "the question text", 
      "candidates": ["candidate answer 1 text", "candidate answer 2 text", ...]
    }
    ```

  * __Success Response:__

    * __Code:__ 200

      __Content:__ 

      ```
      {
        'question': {
            'tokens': ['the', 'question', 'text']
        },
        'answers': [
            {
                'tokens': ['candidate', 'answer', '1', 'text'],
                'weights': [0.1, 0.2, 0.3, 0.4],
                'questionWeights': [0.1, 0.2, 0.7]
            },
            ...
        ]
      }
      ```