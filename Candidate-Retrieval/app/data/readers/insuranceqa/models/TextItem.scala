package data.readers.insuranceqa.models

import play.api.libs.json._

/**
  * Represents a tokenized text
  *
  * @param sentences
  */
case class TextItem(text: String, sentences: List[Sentence]) extends Metadata {

}

case class Sentence(text: String, tokens: List[Token]) extends Metadata {

}

case class Token(text: String) extends Metadata {
  override def toString: String = text
}

//
// JSON Serialization
//

case object Token {
  implicit val tokenWrites = new Writes[Token] {
    def writes(token: Token) = Json.obj(
      "text" -> token.text,
      "metadata" -> Json.toJson(token.metadata.toMap)
    )
  }

  implicit val tokenReads = new Reads[Token] {
    def reads(json: JsValue): JsResult[Token] = {
      val text = (json \ "text").as[String]
      val metadata = (json \ "metadata").as[Map[String, String]]
      val item = Token(text)
      item.metadata ++= metadata
      JsSuccess(item)
    }
  }
}

case object Sentence {
  implicit val sentenceWrites = new Writes[Sentence] {
    def writes(sentence: Sentence) = Json.obj(
      "text" -> sentence.text,
      "tokens" -> Json.toJson(sentence.tokens),
      "metadata" -> Json.toJson(sentence.metadata.toMap)
    )
  }

  implicit val sentenceReads = new Reads[Sentence] {
    def reads(json: JsValue): JsResult[Sentence] = {
      val text = (json \ "text").as[String]
      val tokens = (json \ "tokens").as[List[Token]]
      val metadata = (json \ "metadata").as[Map[String, String]]
      val sentence = Sentence(text, tokens)
      sentence.metadata ++= metadata
      JsSuccess(sentence)
    }
  }
}

case object TextItem {
  implicit val textItemWrites = new Writes[TextItem] {
    def writes(textItem: TextItem) = Json.obj(
      "text" -> textItem.text,
      "sentences" -> Json.toJson(textItem.sentences),
      "metadata" -> Json.toJson(textItem.metadata.toMap)
    )
  }

  implicit val textItemReads = new Reads[TextItem] {
    def reads(json: JsValue): JsResult[TextItem] = {
      val text = (json \ "text").as[String]
      val sentences = (json \ "sentences").as[List[Sentence]]
      val metadata = (json \ "metadata").as[Map[String, String]]
      val item = TextItem(text, sentences)
      item.metadata ++= metadata
      JsSuccess(item)
    }
  }

}