package data.readers

object DataReaderRepository {
  def getReader(name: String): DataReader = name match {
    case "insuranceqav2" | "insuranceqa" => new InsuranceQAV2DataReader()
    case "stackexchange" => new SEDataReader()
    case "insuranceqav1" => new InsuranceQAV1DataReader()

    case _ => new SEDataReader()
  }
}
