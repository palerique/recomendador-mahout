package br.com.sitedoph.machineLearningIntroducaoAosSistemasDeRecomendacoes;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.util.List;

/**
 * Criado por ph em 12/2/16.
 */
public class ModelBasedRecommender {
    private final DataModel          model;
    private final RecommenderBuilder builder;
    private final Recommender        recommender;
    private final double             erro;

    public ModelBasedRecommender(DataModel model) throws TasteException {
        this.model = model;
        this.builder = new RecomendadorBuilder();
        recommender = builder.buildRecommender(model);
        final RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        erro = evaluator.evaluate(builder, null, model, 0.9, 1.0);
    }

    public List<RecommendedItem> getRecommendations(int id, int quantity) throws TasteException {
        return recommender.recommend(id, quantity);
    }

    public double evaluateRecommendationsQuality() throws TasteException {
        return erro;
    }
}
