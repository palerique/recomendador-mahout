package br.com.sitedoph.testes_mahout;

import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Criado por ph em 12/8/16.
 */
public class TestesMahoutApp {

    public static void main(String[] args) {

        //features = cor = [ 1 - branco, 2 - negro, 3 - amarelo, 4 - moreno, 5 - pardo ]
        //features = altura
        //features = peso
        //features = idade
        double[][] featuresArray = {
                {1.0, 150.0, 50.0, 30.0}, //1
                {2.0, 140.0, 55.0, 31.0}, //2
                {3.0, 160.0, 70.0, 32.0}, //3
                {4.0, 170.0, 85.0, 33.0}, //4
                {5.0, 180.0, 105.0, 34.0}, //5
                {1.0, 160.0, 49.0, 35.0}, //6
                {2.0, 190.0, 115.0, 36.0}, //7
                {3.0, 215.0, 130.0, 37.0}, //8
                {4.0, 185.0, 77.0, 38.0}, //9
                {5.0, 181.0, 81.0, 39.0}  //10
        };

        //category = carro = [ 1 - fusca, 2 - journey, 3 - civic, 4 - fusion, 5 - palio ]
        double[] labelOrCategoryWeightArray = {
                1.0,
                2.0,
                3.0,
                4.0,
                5.0
        };
        double[] featureWeightArray = {1.0, 1.0, 1.0, 1.0};

        Matrix featureMatrix = new DenseMatrix(featuresArray);
        Vector labelOrCategoryVector = new DenseVector(labelOrCategoryWeightArray);
        Vector featureWeightVector = new DenseVector(featureWeightArray);

        // now generate the model
        NaiveBayesModel model = new NaiveBayesModel(featureMatrix, featureWeightVector, labelOrCategoryVector, null, 1.0f, true);

        StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(model);

        final Vector probabilityVector = classifier.classifyFull(new DenseVector(new double[]{2.0, 220.0, 100.0, 30.0}));

        int maxIndex = probabilityVector.maxValueIndex();
        System.out.println("category = " + maxIndex + " prob: " + probabilityVector.get(maxIndex));

//        for (Vector.Element element : probabilityVector.all()) {
//            System.out.println(" index: " + (element.index() + 1)  + " element: " + element.get());
//        }
    }

}
