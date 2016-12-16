package br.com.sitedoph.spark_examples;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Criado por ph em 12/14/16.
 */
@Slf4j
public class SparkBasicSessionReadingCSVApp {

    public static final  String[]     FEATURES_STRING_COLUMNS = new String[]{"gender", "time", "uf"};
    public static final  Set<String>  featuresNumericColumns  = new HashSet<>(Collections.singletonList("age"));
    private static final SparkSession spark                   = SparkSession
            .builder()
            .appName("Java Spark SQL basic example")
            .config("spark.master", "local")
            .getOrCreate();

    public static void main(String[] args) {

        Dataset<Row> data = loadAndParseCSV("src/main/resources/people2.csv");
        //data.printSchema();
        //data.show();

        /*
            example:
            +---+------+---+-----------------+-----------+---+
            |age|gender| id|             name|       time| uf|
            +---+------+---+-----------------+-----------+---+
            | 68|  Male|  1|    Carl Matthews|   cruzeiro| MG|
            | 28|  Male|  2|     Adam Johnson|   cruzeiro| MG|
            | 43|Female|  3|Kimberly Mcdonald|  palmeiras| MG|
            | 65|Female|  4|    Teresa Butler|chapecoense| SC|
            | 61|Female|  5|  Michelle Peters|  palmeiras| DF|
         */

        writeToDisk(data);

        data = convertStringColumnsToNumeric(data);
        data = createFeaturesVector(data);

        /*
            +---+------+----+-------------------+-----------+---+--------------+------------+----------+-------------------+
            |age|gender|  id|               name|       time| uf|indexed-gender|indexed-time|indexed-uf|           features|
            +---+------+----+-------------------+-----------+---+--------------+------------+----------+-------------------+
            | 68|  Male|   1|      Carl Matthews|   cruzeiro| MG|           1.0|         3.0|       2.0| [68.0,1.0,3.0,2.0]|
            | 28|  Male|   2|       Adam Johnson|   cruzeiro| MG|           1.0|         3.0|       2.0| [28.0,1.0,3.0,2.0]|
            | 43|Female|   3|  Kimberly Mcdonald|  palmeiras| MG|           0.0|         0.0|       2.0| [43.0,0.0,0.0,2.0]|
            | 65|Female|   4|      Teresa Butler|chapecoense| SC|           0.0|         2.0|       1.0| [65.0,0.0,2.0,1.0]|
            | 61|Female|   5|    Michelle Peters|  palmeiras| DF|           0.0|         0.0|       3.0| [61.0,0.0,0.0,3.0]|
            | 10|Female|   6|     Phyllis Burton|   cruzeiro| GO|           0.0|         3.0|       0.0| [10.0,0.0,3.0,0.0]|
         */

        VectorIndexerModel featureIndexer = indexFeatures(data);

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a GBT model.
        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("indexed-time")
                .setFeaturesCol("indexedFeatures")
                .setMaxIter(10);

        // Chain indexer and GBT in a Pipeline.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{featureIndexer, gbt});

        // Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

        // Make controlledTestPredictions.
        Dataset<Row> controlledTestPredictions = model.transform(testData);

        // Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluatorRMSE = new RegressionEvaluator()
                .setLabelCol("indexed-time")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluatorRMSE.evaluate(controlledTestPredictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        RegressionEvaluator evaluatorMAE = new RegressionEvaluator()
                .setLabelCol("indexed-time")
                .setPredictionCol("prediction")
                .setMetricName("mae");
        double mae = evaluatorMAE.evaluate(controlledTestPredictions);
        System.out.println("Mean Absolute Error (MAE) on test data = " + mae);

        RegressionEvaluator evaluatorMSE = new RegressionEvaluator()
                .setLabelCol("indexed-time")
                .setPredictionCol("prediction")
                .setMetricName("mse");
        double mse = evaluatorMSE.evaluate(controlledTestPredictions);
        System.out.println("(MSE) on test data = " + mse);

        GBTRegressionModel gbtModel = (GBTRegressionModel) (model.stages()[1]);
        System.out.println("Learned regression GBT model:\n" + gbtModel.toDebugString());

        // Select example rows to display.
        controlledTestPredictions.select("id", "name", "gender", "indexed-gender", "age", "uf", "indexed-uf", "time", "indexed-time", "prediction", "features")
                                 .show(300);
        /*
            Root Mean Squared Error (RMSE) on test data = 1.1724134725228308
            comparing the real with the predicted:
            +---+-------------------+------+--------------+---+---+----------+-----------+------------+--------------------+---------------+
            | id|               name|gender|indexed-gender|age| uf|indexed-uf|       time|indexed-time|          prediction|       features|
            +---+-------------------+------+--------------+---+---+----------+-----------+------------+--------------------+---------------+
            |161|        Jason Ortiz|  Male|           1.0|  1| SC|       1.0|     santos|         1.0|  0.9651456444206081|  [1.0,1.0,1.0]|
            |526|     William Hughes|  Male|           1.0|  1| SC|       1.0|chapecoense|         2.0|  0.9651456444206081|  [1.0,1.0,1.0]|
            |465|      Christina Cox|Female|           0.0|  2| MG|       2.0|chapecoense|         2.0|  1.3620207757491984|  [0.0,2.0,2.0]|
            |694|   Walter Armstrong|  Male|           1.0|  2| SC|       1.0|     santos|         1.0|  0.9651456444206081|  [1.0,1.0,2.0]|
            |717|     Benjamin Perez|  Male|           1.0|  2| SC|       1.0|     santos|         1.0|  0.9651456444206081|  [1.0,1.0,2.0]|
            |786|      Roger Freeman|  Male|           1.0|  2| DF|       3.0|   cruzeiro|         3.0|   1.517759840088324|  [1.0,3.0,2.0]|
            |800|     Martin Collins|  Male|           1.0|  2| MG|       2.0|  palmeiras|         0.0|  1.7410534047082629|  [1.0,2.0,2.0]|
            |991|        Daniel Ward|  Male|           1.0|  2| GO|       0.0|chapecoense|         2.0|  0.6845786549541729|  [1.0,0.0,2.0]|
            | 12|      Matthew Mccoy|  Male|           1.0|  3| MG|       2.0|  palmeiras|         0.0|  1.4357731028323633|  [1.0,2.0,3.0]|
            |360|      Brandon Ramos|  Male|           1.0|  3| DF|       3.0|chapecoense|         2.0|  1.9015427874504471|  [1.0,3.0,3.0]|
            |523|          John King|  Male|           1.0|  3| MG|       2.0|  palmeiras|         0.0|  1.4357731028323633|  [1.0,2.0,3.0]|
            |819|        Bobby Lynch|  Male|           1.0|  3| DF|       3.0|chapecoense|         2.0|  1.9015427874504471|  [1.0,3.0,3.0]|
         */

//        //TODO: Making predictions with unknown data:
//
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//        log.info("%%%%%%%%%%%    Starting Predictions With Unknown Data    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//
//        Dataset<Row> semTimeDS = loadAndParseCSV("src/main/resources/people2-sem-time.csv");
////        semTimeDS = convertStringColumnsToNumeric(semTimeDS);
////        semTimeDS = createFeaturesVector(semTimeDS);
//        semTimeDS.show(100);
//
//        final Dataset<Row> semTimePredictedDS = model.transform(semTimeDS);
//
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//
//        semTimePredictedDS.show(100);
//
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
//        log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
    }

    private static Dataset<Row> loadAndParseCSV(String path) {
        // Load and parse the data file
        final JavaRDD<Person> personJavaRDD = spark.read()
                                                   .option("header", "true")
                                                   .csv(path)
                                                   .javaRDD()
                                                   .map((Function<Row, Person>) row -> Person.builder()
                                                                                             .id(Long.parseLong(row.getString(0)))
                                                                                             .name(row.getString(1))
                                                                                             .gender(row.getString(2))
                                                                                             .age(Integer.parseInt(row.getString(3)))
                                                                                             .time(row.getString(4))
                                                                                             .uf(row.getString(5))
                                                                                             .build());

        return spark.createDataFrame(personJavaRDD, Person.class);
    }

    private static void writeToDisk(Dataset<Row> data) {
//        data.write().json("src/main/resources/data-people.json");
//        data.write().format("libsvm").save("src/main/resources/data-people-libsvm.txt");
    }

    private static Dataset<Row> convertStringColumnsToNumeric(Dataset<Row> data) {
        // Indexing is done to improve the execution times as comparing indexes
        // is much cheaper than comparing strings/floats

        for (String column : FEATURES_STRING_COLUMNS) {
            final String outputCol = "indexed-" + column;
            featuresNumericColumns.add(outputCol);
            StringIndexer indexer = new StringIndexer()
                    .setInputCol(column)
                    .setOutputCol(outputCol);
            data = indexer.fit(data).transform(data);
        }
        return data;
    }

    private static Dataset<Row> createFeaturesVector(Dataset<Row> data) {
        // Create a vector from columns. Name the resulting vector as "features"
        VectorAssembler vectorAssembler = new VectorAssembler();
        String[] featuresWithoutTeam = new String[featuresNumericColumns.size() - 1];
        int i = 0;
        for (String column : featuresNumericColumns) {
            if (!column.equals("indexed-time")) {
                featuresWithoutTeam[i] = column;
                i++;
            }
        }
        vectorAssembler.setInputCols(featuresWithoutTeam).setOutputCol("features");
        data = vectorAssembler.transform(data);
        return data;
    }

    private static VectorIndexerModel indexFeatures(Dataset<Row> data) {
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        return new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);
    }

}
