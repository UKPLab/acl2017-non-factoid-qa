package data.readers.insuranceqa.models

import scala.collection.mutable

trait Metadata {
  val metadata: mutable.Map[String, String] = mutable.Map.empty[String, String]
}
