logLevel := Level.Warn

resolvers += "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/"

// newer versions don't work together with retrieval.elastic
addSbtPlugin("com.typesafe.play" % "sbt-plugin" % "2.4.8")