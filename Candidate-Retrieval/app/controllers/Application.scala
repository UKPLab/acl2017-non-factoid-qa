package controllers

import javax.inject.Inject

import data.ElasticConnector
import play.api.Configuration
import play.api.libs.json.{JsArray, JsObject, JsString}
import play.api.mvc._
import data.readers.{DataReaderRepository}
import data.writers.TSVArchiveWriter
import scala.collection.mutable
import play.api.Logger

class Application @Inject()(configuration: play.api.Configuration) extends Controller {

  lazy val indexName = configuration.getString("retrieval.elastic.indexName").get
  lazy val elasticHost = configuration.getString("retrieval.elastic.host").get
  lazy val elasticPort = configuration.getInt("retrieval.elastic.port").get
  lazy val elastic = new ElasticConnector(elasticHost, elasticPort, indexName)

  def index = Action {
    Ok("The application is running")
  }

  def createIndex = Action {
    val reader = DataReaderRepository.getReader(configuration.getString("retrieval.dataset.type").get)
    val datasetPath = configuration.getString("retrieval.dataset.path").get
    val datasetOptions = configuration.getConfig("retrieval.dataset.options").getOrElse(Configuration.empty)

    val answers = reader.readAnswers(datasetPath, datasetOptions)

    elastic.createIndex(true)
    elastic.saveAnswers(answers)

    Ok("Index created")
  }

  def queryIndex(q: String, n: Int) = Action {
    Ok(
      JsObject(Seq(
        "candidates" -> JsArray(elastic.queryAnswers(q, n).map(a => JsString(a.text)))
      ))
    )
  }

  def writeTSVArchive = Action {
    // TODO use a streaming response instead of log messages
    Logger.info("Writing TSV Archive")
    val datasetPath = configuration.getString("retrieval.dataset.path").get
    val datasetOptions = configuration.getConfig("retrieval.dataset.options").getOrElse(Configuration.empty)

    Logger.info("Reading source dataset")
    val reader = DataReaderRepository.getReader(configuration.getString("retrieval.dataset.type").get)


    val questionTexts = mutable.Map[String, String]()
    val answerTexts = mutable.Map[String, String]()
    val qas = reader.readQA(datasetPath, datasetOptions)
    val qasMap = qas.map(qa => qa.question.id -> qa).toMap

    // construct all the pools
    Logger.info("Constructing pools")
    val pools = qas.zipWithIndex.map { case (qa, idx) =>
      if(idx % 100 == 0) {
        Logger.info(s"$idx/${qas.length}")
      }
      val questionText = qa.question.text.take(5000)
      val pool = elastic.queryAnswers(questionText, 100)
      questionTexts(qa.question.id) = questionText
      (pool ++ qa.groundTruth).foreach { a =>
        answerTexts(a.id) = a.text
      }
      (qa.question.id, qa.groundTruth.map(_.id), pool.map(_.id))
    }

    // write the files
    val trainSize = configuration.getDouble("tsvWriter.split.train").get
    val validSize = configuration.getDouble("tsvWriter.split.valid").get

    Logger.info("Writing archive")
    val writer = new TSVArchiveWriter(configuration.getString("tsvWriter.path").get)
    writer.write(questionTexts.toMap, answerTexts.toMap, pools, trainSize, validSize)

    Logger.info("Done")
    Ok("Dataset created")
  }

}