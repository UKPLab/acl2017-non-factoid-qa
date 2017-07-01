package data.writers

import java.io.{File, FileOutputStream, OutputStreamWriter}
import java.util.zip.GZIPOutputStream

import data.writers.preprocessing.StanfordPreprocessor

import scala.collection.mutable

class TSVArchiveWriter(folderPath: String) {
  private val folderFile = new File(folderPath)

  /**
    * Writes a TSV archive file
    *
    * @param questions questionId -> question text
    * @param answers   answerId -> answer text
    * @param pools     questionId, groundTruthAnswerIds, pooledAnswerIds
    * @param trainSize size of the training split
    * @param validSize size of the valid split
    */
  def write(questions: Map[String, String], answers: Map[String, String], pools: Seq[(String, Seq[String], Seq[String])],
            trainSize: Double, validSize: Double) {

    val preprocessor = new StanfordPreprocessor()

    val (preprocessedQuestions, vocabQuestions) = preprocess(preprocessor, questions)
    val (preprocessedAnswers, vocabAnswers) = preprocess(preprocessor, answers)

    val vocab = vocabQuestions ++ vocabAnswers
    val vocabMap = vocab.zipWithIndex.toMap
    writeVocab(vocabMap)


    val itemsTrain = (trainSize * pools.size).toInt
    val itemsValid = (validSize * pools.size).toInt

    val trainPool = pools.take(itemsTrain)
    val validPool = pools.slice(itemsTrain, itemsTrain + itemsValid)
    val testPool = pools.drop(itemsTrain + itemsValid)

    writeSplit("train", trainPool)
    writeSplit("valid", validPool)
    writeSplit("test", testPool)

    writeTextItems("questions", preprocessedQuestions, vocabMap)
    writeTextItems("answers", preprocessedAnswers, vocabMap)
  }

  private def preprocess(preprocessor: StanfordPreprocessor, items: Map[String, String]) = {
    val vocab = mutable.Set[String]()
    val preprocessedItems = items.map { case (id, text) =>
      val tokens = preprocessor.preprocess(text)
      vocab ++= tokens.toSet
      id -> tokens
    }

    preprocessedItems.toMap -> vocab
  }

  private def writeVocab(vocab: Map[String, Int]): Unit = {
    val output = new FileOutputStream(s"$folderFile/vocab.tsv.gz")
    val writer = new OutputStreamWriter(new GZIPOutputStream(output), "UTF-8")
    vocab.foreach { case (entry, id) =>
      writer.write(s"$id\t$entry\n")
    }
    writer.close()
  }

  private def writeSplit(name:String, pools:Seq[(String, Seq[String], Seq[String])]) {
    val output = new FileOutputStream(s"$folderFile/$name.tsv.gz")
    val writer = new OutputStreamWriter(new GZIPOutputStream(output), "UTF-8")

    pools.foreach { case (questionId, groundTruthIds, answersIds) =>
      writer.write(questionId)
      writer.write("\t")
      writer.write(groundTruthIds.mkString(" "))
      writer.write("\t")
      writer.write(answersIds.mkString(" "))
      writer.write("\n")
    }

    writer.close()
  }

  private def writeTextItems(name: String, items: Map[String, Seq[String]], vocab: Map[String, Int]) {
    val output = new FileOutputStream(s"$folderFile/$name.tsv.gz")
    val writer = new OutputStreamWriter(new GZIPOutputStream(output), "UTF-8")

    items.foreach { case (id, tokens) =>
      writer.write(id)
      writer.write("\t")
      writer.write(tokens.map(t => vocab(t).toString).mkString(" "))
      writer.write("\n")
    }

    writer.close()
  }
}
