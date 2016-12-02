package br.com.sitedoph;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class App {

    private final File file;

    public static void main(String[] args) throws IOException, TasteException {
        DataModel model = new FileDataModel(new App("dados.csv").getFile());
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
        UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        final List<RecommendedItem> recommendedItems = recommender.recommend(2, 3);

        for (RecommendedItem recommendedItem : recommendedItems) {
            System.out.println(recommendedItem);
        }

    }

    public App(String filename) {
        ClassLoader classLoader = getClass().getClassLoader();
        file = new File(classLoader.getResource(filename).getFile());
    }

    public File getFile() {
        return file;
    }
}
