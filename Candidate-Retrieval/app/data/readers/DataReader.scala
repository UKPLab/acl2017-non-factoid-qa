package data.readers

import data.readers.insuranceqa.{ArchiveReader, ArchiveReaderV1, ArchiveReaderV2}
import data.readers.stackexchange.{SEAnswer, SEXMLArchiveReader}
import play.api.Configuration
import play.api.libs.json._

trait DataReader {
  def readQA(path: String, configOptions: Configuration): Seq[QA]

  def readAnswers(path: String, configOptions: Configuration): Seq[TextItem]
}

case class QA(question: TextItem, groundTruth: List[TextItem])

case class TextItem(id: String, text: String)

object TextItem {
  implicit val textItemReads = Json.reads[TextItem]
  implicit val textItemWrites = Json.writes[TextItem]
}


//
// Actual data readers based on the data.readers
//

class SEDataReader extends DataReader {
  override def readQA(path: String, configOptions: Configuration): Seq[QA] = {
    val archive = new SEXMLArchiveReader().read(path)
    val minScore = configOptions.getInt("minScore").getOrElse(1)
    val threads = archive.threads
    val threadsMap = archive.threads.map(t => t._1.id -> t).toMap

    threads
      .filter(_._1.metadata.getOrElse("Score", "0").toInt > minScore)
      .map { case (question, answers) =>

        val duplicates = question.metadata.getOrElse("DuplicateQuestions", "").split(",")
        val duplicatesGroundTruth = duplicates.map(threadsMap.get)
          .filter(_.isDefined)
          .flatMap(t => t.get._2.filter(_.metadata.getOrElse("Score", "0").toInt > 0))
          .toSeq

        val groundTruth = answers.filter(_.metadata.getOrElse("Score", "0").toInt > minScore)

        QA(
          TextItem(question.id, question.text),
          (groundTruth ++ duplicatesGroundTruth).map(a => TextItem(a.id, a.text)).toList
        )
      }.toSeq
  }

  override def readAnswers(path: String, configOptions: Configuration): Seq[TextItem] = {
    val archive = new SEXMLArchiveReader().read(path)
    val minScore = configOptions.getInt("minScore").getOrElse(1)
    archive.threads
      .filter(_._1.metadata.getOrElse("Score", "0").toInt > minScore)
      .values
      .flatten
      .filter(_.metadata.getOrElse("Score", "0").toInt > minScore)
      .toSeq
      .map(a => TextItem(a.id, a.text))
  }
}


trait InsuranceQADataReader extends DataReader {
  def getArchiveReader(path: String, configOptions: Configuration): ArchiveReader

  override def readQA(path: String, configOptions: Configuration): Seq[QA] = {
    val archive = getArchiveReader(path, configOptions).read
    (archive.train.qa ++ archive.valid.qa ++ archive.test.qa).map { pool =>
      QA(
        TextItem(pool.question.metadata("id"), pool.question.text),
        pool.groundTruth.map(a => TextItem(a.metadata("id"), a.text))
      )
    }
  }

  override def readAnswers(path: String, configOptions: Configuration): Seq[TextItem] = {
    getArchiveReader(path, configOptions).read.answers.map(a => TextItem(a.metadata("id"), a.text))
  }
}


class InsuranceQAV1DataReader extends InsuranceQADataReader {
  override def getArchiveReader(path: String, configOptions: Configuration): ArchiveReader = {
    new ArchiveReaderV1(path)
  }
}

class InsuranceQAV2DataReader extends InsuranceQADataReader {
  override def getArchiveReader(path: String, configOptions: Configuration): ArchiveReader = {
    val poolsize = configOptions.getInt("pooledAnswers").getOrElse(500)
    val tokenizer = configOptions.getString("tokenizer").getOrElse("token")
    new ArchiveReaderV2(path, poolsize, tokenizer)
  }
}