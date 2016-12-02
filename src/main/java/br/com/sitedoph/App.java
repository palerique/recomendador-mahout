package br.com.sitedoph;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.RandomUtils;

import java.io.IOException;
import java.util.List;

public class App {

    public static void main(String[] args) throws IOException, TasteException {
        RandomUtils.useTestSeed();
        System.out.println(">*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<");
        doRecomendationsTo("dados.csv", 2, 3);
        System.out.println(">*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<");
        doRecomendationsTo("cursos.csv", 2, 3);
        System.out.println(">*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<");
    }

    private static void doRecomendationsTo(String path, int id, int quantity) throws IOException, TasteException {
        final DataModel model = new FileDataModelBuilder(path).getDataModel();
        final ModelBasedRecommender modelBasedRecommender = new ModelBasedRecommender(model);
        System.out.println("Recomendations quality to file " + path + " is: " + modelBasedRecommender.evaluateRecommendationsQuality());
        final List<RecommendedItem> recommendedItems = modelBasedRecommender.getRecommendations(id, quantity);
        for (RecommendedItem recommendedItem : recommendedItems) {
            System.out.println(recommendedItem);
        }
    }
}
