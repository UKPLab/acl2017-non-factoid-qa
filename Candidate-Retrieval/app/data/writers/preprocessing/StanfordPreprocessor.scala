package data.writers.preprocessing

import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Applies tokenization using stanford corenlp
  */
class StanfordPreprocessor {
  private val props = new Properties()
  props.setProperty("annotators", "tokenize")
  props.setProperty("tokenize.options", "untokenizable=noneDelete,normalizeAmpersandEntity=false," +
    "normalizeParentheses=false,normalizeOtherBrackets=false,asciiQuotes=true")

  private var pipeline = new StanfordCoreNLP(props)

  def preprocess(text: String): List[String] = {
    val result = mutable.MutableList[String]()

    val annotation = new Annotation(text)
    pipeline.annotate(annotation)
    val annotationSentences = annotation.get(classOf[TokensAnnotation]).asScala

      for (stanfordToken <- annotation.get(classOf[TokensAnnotation]).asScala) {
        val tokenText = stanfordToken.originalText()
        if(tokenText.length > 0) {
          result += tokenText
        }
      }

    result.toList
  }
}
