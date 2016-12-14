package br.com.sitedoph.spark_examples;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Criado por ph em 12/14/16.
 */
public class SparkBasicSessionReadingCSVApp {

    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config("spark.master", "local")
                .getOrCreate();


        // Load and parse the data file
        JavaRDD<Person> peopleRDD = spark.read()
                                         .option("header", "true")
                                         .csv("src/main/resources/people2.csv")
                                         .javaRDD()
                                         .map(new Function<Row, Person>() {
                                             @Override
                                             public Person call(Row row) throws Exception {

                                                 System.out.println(row);

                                                 Person person = new Person();
                                                 person.setName(row.getString(0));
                                                 person.setAge(Integer.parseInt(row.getString(1)));
                                                 return person;
                                             }
                                         });

        // Apply a schema to an RDD of JavaBeans to get a DataFrame
        Dataset<Row> data = spark.createDataFrame(peopleRDD, Person.class);

        data.show();

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a GBT model.
        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("label")
                .setFeaturesCol("indexedFeatures")
                .setMaxIter(10);

        // Chain indexer and GBT in a Pipeline.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{featureIndexer, gbt});

        // Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5);

        // Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        GBTRegressionModel gbtModel = (GBTRegressionModel) (model.stages()[1]);
        System.out.println("Learned regression GBT model:\n" + gbtModel.toDebugString());
    }

}
