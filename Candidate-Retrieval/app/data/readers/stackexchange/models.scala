package data.readers.stackexchange

import scala.collection.mutable

trait Metadata {
  val metadata: mutable.Map[String, String] = mutable.Map.empty[String, String]
}


trait SETextItem extends Metadata {
  val id: String
}

case class SEQuestion(id: String, title: String, text: String) extends SETextItem

case class SEAnswer(id: String, text: String) extends SETextItem


case class SEArchive(threads:Map[SEQuestion, Seq[SEAnswer]])