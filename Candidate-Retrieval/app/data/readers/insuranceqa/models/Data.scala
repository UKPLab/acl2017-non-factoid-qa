package data.readers.insuranceqa.models

/**
  * A data split such as train, valid, test
  *
  * @param splitName e.g. train
  * @param qa the individual data points
  */
case class Data(splitName:String, qa:List[QAPool]) {

}
