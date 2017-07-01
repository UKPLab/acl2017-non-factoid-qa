package data.readers.insuranceqa

import java.io.InputStream

import data.readers.insuranceqa.models.Archive

import scala.io.Source

/**
  * Defines the basic interface for any archive reader (v1, v2, ...)
  */
trait ArchiveReader {
  def read: Archive

  protected def readTsv(is: InputStream) = {
    val s = Source.fromInputStream(is)
    val result = s.getLines().filterNot(_.isEmpty).map { l =>
      l.split("\t")
    }.toList
    s.close
    result
  }
}