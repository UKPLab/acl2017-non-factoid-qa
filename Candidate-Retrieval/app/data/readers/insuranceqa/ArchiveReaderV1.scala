package data.readers.insuranceqa

import java.io.FileInputStream

import data.readers.insuranceqa.models._

/**
  * Allows to read the insuranceQA V1 files
  *
  * @param insuranceQAPath root path of the insurance dataset (contains V2 folder)
  */
class ArchiveReaderV1(insuranceQAPath: String) extends ArchiveReader {

  protected def filePath(filename: String) = s"$insuranceQAPath/V1/$filename"

  protected def readSplit(name: String, vocab: Map[String, String], answers: Map[String, TextItem]): Data = {
    val train = name equals "train"
    val filename = if (train) {
      "question.train.token_idx.label"
    } else {
      s"question.$name.label.token_idx.pool"
    }
    val dataPoints = readTsv(new FileInputStream(filePath(filename))).zipWithIndex.map { case (line, idx) =>
      val questionLine = if (train) line(0) else line(1)
      val groundTruthLine = if (train) line(1) else line(0)
      val poolLine = if (train) None else Some(line(2))

      val groundTruth = groundTruthLine.split(" ").map(answers(_))
      val pool = if (poolLine.isDefined) poolLine.get.split(" ").map(answers(_)).toList else List.empty[TextItem]

      val questionTokens = questionLine.split(" ").map(t => Token(vocab(t))).toList
      val questionSentence = Sentence(questionTokens.mkString(" "), questionTokens)
      val question = TextItem(questionSentence.text, List(questionSentence))
      question.metadata("id") = s"$name-$idx"

      QAPool(question, pool, groundTruth.toList)
    }
    Data(name, dataPoints)
  }

  override def read: Archive = {
    val vocab = readTsv(new FileInputStream(filePath("vocabulary"))).map(l => l.head -> l(1)).toMap
    val answers = readTsv(new FileInputStream(filePath(s"answers.label.token_idx"))).map { line =>
      val id = line.head
      val tokens = line(1).split(" ").map(t => Token(vocab(t))).toList
      val answerSentence = Sentence(tokens.mkString(" "), tokens)
      val answer = TextItem(answerSentence.text, List(answerSentence))
      answer.metadata("id") = id
      id -> answer
    }.toMap

    val train = readSplit("train", vocab, answers)
    val valid = readSplit("dev", vocab, answers)
    val test = readSplit("test1", vocab, answers)

    val questions = (train.qa ++ valid.qa ++ test.qa).map(_.question)

    Archive(train, valid, test, questions, answers.values.toList)
  }

}
