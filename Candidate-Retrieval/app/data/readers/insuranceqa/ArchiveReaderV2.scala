package data.readers.insuranceqa

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import data.readers.insuranceqa.models._

/**
  * Allows to read the insuranceQA V2 files
  *
  * @param insuranceQAPath root path of the insurance dataset (contains V2 folder)
  * @param pooledAnswers   the answer-pool size (100, 500, 1000, 1500)
  * @param tokenizer       the tokenizer type (v2 has either 'raw' or 'token')
  */
class ArchiveReaderV2(insuranceQAPath: String, pooledAnswers: Int = 500, tokenizer: String = "token") extends ArchiveReader {

  protected def filePath(filename: String) = s"$insuranceQAPath/V2/$filename"

  protected def gzInputStream(path: String) = new GZIPInputStream(new BufferedInputStream(new FileInputStream(path)))

  protected def readSplit(name: String, vocab: Map[String, String], answers: Map[String, TextItem]): Data = {
    val filename = s"InsuranceQA.question.anslabel.$tokenizer.$pooledAnswers.pool.solr.$name.encoded.gz"
    val dataPoints = readTsv(gzInputStream(filePath(filename))).zipWithIndex.map { case (line, idx) =>
      val questionTokens = line(1).split(" ").map(t => Token(vocab(t))).toList
      val questionSentence = Sentence(questionTokens.mkString(" "), questionTokens)
      val question = TextItem(questionSentence.text, List(questionSentence))
      question.metadata("id") = s"$name-$idx"
      val groundTruth = line(2).split(" ").map(answers(_)).toList
      val pool = line(3).split(" ").map(answers(_)).toList
      QAPool(question, pool, groundTruth)
    }
    Data(name, dataPoints)
  }

  override def read: Archive = {
    val vocab = readTsv(new FileInputStream(filePath("vocabulary"))).map(l => l.head -> l(1)).toMap
    val answers = readTsv(gzInputStream(filePath(s"InsuranceQA.label2answer.$tokenizer.encoded.gz"))).map { line =>
      val id = line.head
      val tokens = line(1).split(" ").map(t => Token(vocab(t))).toList
      val answerSentence = Sentence(tokens.mkString(" "), tokens)
      val answer = TextItem(answerSentence.text, List(answerSentence))
      answer.metadata("id") = id
      id -> answer
    }.toMap

    val train = readSplit("train", vocab, answers)
    val valid = readSplit("valid", vocab, answers)
    val test = readSplit("test", vocab, answers)

    val questions = (train.qa ++ valid.qa ++ test.qa).map(_.question)

    Archive(train, valid, test, questions, answers.values.toList)
  }

}
