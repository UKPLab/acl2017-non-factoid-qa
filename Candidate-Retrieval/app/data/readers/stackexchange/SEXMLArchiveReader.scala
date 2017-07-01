package data.readers.stackexchange

import javax.xml.parsers.{SAXParser, SAXParserFactory}

import org.xml.sax.Attributes
import org.xml.sax.helpers.DefaultHandler

import scala.collection.mutable


class SEXMLArchiveReader {
  def read(path: String): SEArchive = {
    val factory: SAXParserFactory = SAXParserFactory.newInstance
    val saxParser: SAXParser = factory.newSAXParser

    val postsHandler: PostsHandler = new PostsHandler()
    saxParser.parse(s"$path/Posts.xml", postsHandler)
    val (questions, threads) = postsHandler.result

    val handler = new PostLinkHandler(questions)
    saxParser.parse(s"$path/PostLinks.xml", handler)

    SEArchive(threads)
  }
}


class PostsHandler extends DefaultHandler {
  val questions = new mutable.HashMap[String, SEQuestion]
  val threads = new mutable.HashMap[SEQuestion, Seq[SEAnswer]].withDefaultValue(Seq.empty)
  val answersQuestionMissing = new mutable.HashMap[String, SEAnswer]

  override def startElement(uri: String, localName: String, qName: String, attributes: Attributes) {
    if (qName == "row") {
      val id = attributes.getValue("Id")
      val title = attributes.getValue("Title")
      val text = attributes.getValue("Body").replaceAll("\\<.*?>", " ").replaceAll("\n", " ")

      val postTypeId = attributes.getValue("PostTypeId")
      if (postTypeId == "1") {
        // Question
        val textItem = SEQuestion(id, title, text)
        addAttributesToMetadata(attributes, textItem)
        questions.put(id, textItem)
      } else if (postTypeId == "2") {
        // Answer
        val textItem = SEAnswer(id, text)
        addAttributesToMetadata(attributes, textItem)

        val parentId = attributes.getValue("ParentId")
        questions.get(parentId) match {
          case Some(question) =>
            threads(question) = threads(question) :+ textItem
          case None =>
            answersQuestionMissing.put(parentId, textItem)
        }
      }
    }
  }

  def result = {
    answersQuestionMissing.foreach { case (questionId, answer) =>
      questions.get(questionId).foreach { question =>
        threads(question) = threads(question) :+ answer
      }
    }
    answersQuestionMissing.clear()
    questions.toMap -> threads.toMap
  }

  private def addAttributesToMetadata(attributes: Attributes, metadataItem: Metadata) {
    val excludedAttrs = Set("Id", "PostTypeId", "ParentId", "PostId", "Body", "Title", "Text")
    for (i <- 0 to attributes.getLength) {
      val attr = attributes.getQName(i)
      if (!excludedAttrs.contains(attr)) {
        metadataItem.metadata.put(attr, attributes.getValue(i))
      }
    }
  }
}


class PostLinkHandler(var questions: Map[String, SEQuestion]) extends DefaultHandler {
  private var q1Id = null
  private var q2Id = null
  private var q1 = null
  private var q2 = null
  private val addedRelations = mutable.HashSet[(String, String, String)]()

  override def startElement(uri: String, localName: String, qName: String, attributes: Attributes) {
    if (qName == "row") {
      val q1Id = attributes.getValue("PostId")
      val q2Id = attributes.getValue("RelatedPostId")
      val linkTypeID = attributes.getValue("LinkTypeId")

      val q1o = questions.get(q1Id)
      val q2o = questions.get(q2Id)

      val relationType = if (linkTypeID == "1") "RelatedQuestions" else "DuplicateQuestions"
      q1o -> q2o match {
        case (Some(q1), Some(q2)) =>
          if (!addedRelations.contains((q1.id, q2.id, relationType))) {
            q1.metadata(relationType) = s"${q1.metadata.getOrElse(relationType, "")},${q2.id}"
            addedRelations += ((q1.id, q2.id, relationType))
          }
          if (!addedRelations.contains((q2.id, q1.id, relationType))) {
            q2.metadata(relationType) = s"${q2.metadata.getOrElse(relationType, "")},${q1.id}"
            addedRelations += ((q2.id, q1.id, relationType))
          }

        case _ =>
      }
    }
  }
}