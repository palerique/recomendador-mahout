package br.com.sitedoph;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.common.RandomUtils;

import java.io.File;
import java.io.IOException;

/**
 * Criado por ph em 12/2/16.
 */
public class Avaliador {

    private final File file;

    public static void main(String[] args) throws IOException, TasteException {

        RandomUtils.useTestSeed();

        DataModel model = new FileDataModel(new Avaliador("dados.csv").getFile());

        final RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        final RecommenderBuilder builder = new RecomendadorDeProdutosBuilder();

        final double erro = evaluator.evaluate(builder, null, model, 0.9, 1.0);
        System.out.println(erro);
    }

    public Avaliador(String filename) {
        ClassLoader classLoader = getClass().getClassLoader();
        file = new File(classLoader.getResource(filename).getFile());
    }

    public File getFile() {
        return file;
    }
}
