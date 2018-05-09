name := "Candidate-Retrieval"

version := "1.0"

lazy val `candidate-retrieval` = (project in file(".")).enablePlugins(PlayScala)

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(jdbc, cache, ws, specs2 % Test)

unmanagedResourceDirectories in Test <+= baseDirectory(_ / "target/web/public/test")

resolvers += "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases"

routesGenerator := InjectedRoutesGenerator

// our own dependencies

val jackson2Version = "2.7.4"

libraryDependencies ++= Seq(
  "com.google.guava" % "guava" % "20.0",
  "org.elasticsearch.client" % "transport" % "6.2.4",
  "com.fasterxml.jackson.core" % "jackson-core" % jackson2Version,
  "com.fasterxml.jackson.core" % "jackson-databind" % jackson2Version,
  "com.fasterxml.jackson.core" % "jackson-annotations" % jackson2Version,
  "org.apache.logging.log4j" % "log4j-api" % "2.7",
  "org.apache.logging.log4j" % "log4j-core" % "2.7")

libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.5.2"