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
public class TestesMahoutRiscosEDiasApp {

    public static void main(String[] args) {

        double[][] featuresArray = {
                {50},
                {100},
                {150},
                {200},
                {250},
                {300},
                {350},
                {400},
                {450},
                {500}
        };

        double[] labelOrCategoryWeightArray = {1.0};
        double[] featureWeightArray = {1.0};

        Matrix featureMatrix = new DenseMatrix(featuresArray);
        Vector labelOrCategoryVector = new DenseVector(labelOrCategoryWeightArray);
        Vector featureWeightVector = new DenseVector(featureWeightArray);

        // now generate the model
        NaiveBayesModel model = new NaiveBayesModel(featureMatrix, featureWeightVector, labelOrCategoryVector, null, 1.0f, true);

        StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(model);

        final double scoreForLabelFeature = classifier.getScoreForLabelFeature(0, 0);
        System.out.println("scoreForLabelFeature 0x0 : " + scoreForLabelFeature);

        final Vector probabilityVector = classifier.classifyFull(new DenseVector(new double[]{260}));

        int maxIndex = probabilityVector.maxValueIndex();
        System.out.println("category = " + maxIndex + " prob: " + probabilityVector.get(maxIndex));

//        for (Vector.Element element : probabilityVector.all()) {
//            System.out.println(" index: " + (element.index() + 1)  + " element: " + element.get());
//        }
    }

}
