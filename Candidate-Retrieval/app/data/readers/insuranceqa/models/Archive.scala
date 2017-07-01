package data.readers.insuranceqa.models

import scala.collection.mutable

/**
  * The whole InsuranceQA Archive with its individual data splits
  *
  * @param train
  * @param valid
  * @param test
  */
case class Archive(train: Data, valid: Data, test: Data, questions: List[TextItem], answers: List[TextItem]) {
  def getVocab = {
    val vocab = mutable.Set[String]()
    (questions ++ answers).foreach { q =>
      q.sentences.foreach { s =>
        s.tokens.foreach { t =>
          vocab += t.text
        }
      }
    }
    vocab.toSet
  }
}
