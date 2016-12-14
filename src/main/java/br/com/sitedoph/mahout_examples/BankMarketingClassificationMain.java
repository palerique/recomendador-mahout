package br.com.sitedoph.mahout_examples;
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.google.common.collect.Lists;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Uses the SGD classifier on the 'Bank marketing' dataset from UCI.
 * <p>
 * See http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
 * <p>
 * Learn when people accept or reject an offer from the bank via telephone based on income, age, education and more.
 */
public class BankMarketingClassificationMain {

    public static final int NUM_CATEGORIES = 2;

    public static void main(String[] args) throws Exception {
        List<TelephoneCall> calls = Lists.newArrayList(new TelephoneCallParser("bank-full.csv"));

        double heldOutPercentage = 0.10;

        double biggestScore = 0.0;

        for (int run = 0; run < 20; run++) {
            Collections.shuffle(calls);
            int cutoff = (int) (heldOutPercentage * calls.size());
            List<TelephoneCall> testAccuracyData = calls.subList(0, cutoff);
            List<TelephoneCall> trainData = calls.subList(cutoff, calls.size());

            List<TelephoneCall> testUnknownData = new ArrayList<>();

            testUnknownData.add(getUnknownTelephoneCall(trainData));

            OnlineLogisticRegression lr = new OnlineLogisticRegression(NUM_CATEGORIES, TelephoneCall.FEATURES, new L1())
                    .learningRate(1)
                    .alpha(1)
                    .lambda(0.000001)
                    .stepOffset(10000)
                    .decayExponent(0.2);

            for (int pass = 0; pass < 20; pass++) {
                for (TelephoneCall observation : trainData) {
                    lr.train(observation.getTarget(), observation.asVector());
                }
                Auc eval = new Auc(0.5);
                for (TelephoneCall testCall : testAccuracyData) {
                    biggestScore = evaluateTheCallAndGetBiggestScore(biggestScore, lr, eval, testCall);
                }
                System.out.printf("run: %-5d pass: %-5d current learning rate: %-5.4f \teval auc %-5.4f\n", run, pass, lr.currentLearningRate(), eval.auc());

                for (TelephoneCall testCall : testUnknownData) {
                    final double score = lr.classifyScalar(testCall.asVector());
                    System.out.println(" score: " + score + " accuracy " + eval.auc() + " call fields: " + testCall.getFields());
                }
            }
        }
    }

    private static TelephoneCall getUnknownTelephoneCall(List<TelephoneCall> trainData) {
        TelephoneCall example = trainData.get(0);

        List<String> values = new ArrayList<>();

        values.add("54");
        values.add("technician");
        values.add("divorced");
        values.add("postgraduate");
        values.add("no");
        values.add("1904");
        values.add("yes");
        values.add("yes");
        values.add("unknown");
        values.add("7");
        values.add("may");
        values.add("280");
        values.add("3");
        values.add("-1");
        values.add("0");
        values.add("unknown");
        values.add("unknown");

        return new TelephoneCall(example.getFieldNames(), values);
    }

    private static double evaluateTheCallAndGetBiggestScore(double biggestScore, OnlineLogisticRegression lr, Auc eval, TelephoneCall call) {
        final double score = lr.classifyScalar(call.asVector());
        eval.add(call.getTarget(), score);
        if (score > biggestScore) {
            System.out.println("### SCORE > BIGGESTSCORE ### score: " + score + " accuracy " + eval.auc() + " call fields: " + call.getFields());
            biggestScore = score;
        }
        return biggestScore;
    }
}
