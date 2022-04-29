package org.njit.cloudcomputing;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

public class QualityTraining {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			String inputfile = args[0];
			//"/Users/vivekbaskar/eclipse-workspace/QualityPredictionEngine/TrainingDataset.csv";
			SparkSession spark = SparkSession.
					builder()
					.master(args[1])
					.appName("Wine_Quality_Training").getOrCreate();

			JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
			jsc.setLogLevel("ERROR");

			Dataset<Row> input = spark.read()
					.option("header", true)
					.option("inferSchema",true)
					.option("delimiter",",")
					.format("csv")
					.load(inputfile);

			input.show(10,false);


			Dataset<Row> label =input.withColumnRenamed("quality","label");
			label.show(10,false);

			VectorAssembler assembler =new VectorAssembler()
					.setInputCols(new String[] {"fixed acidity","volatile acidity","citric acid","residual sugar"
							,"chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"})
					.setOutputCol("features");


			Dataset<Row> features= assembler.transform(label).select("label","features");

			features=features.withColumn("label",functions.col("label").cast(DataTypes.DoubleType));
			features.show(10,false);

			System.out.println("\n============================LinearRegressionModel Start============================");
			LinearRegression linear = new LinearRegression();

			linear
			.setMaxIter(200)
			.setRegParam(0.3)
			.setElasticNetParam(0.2);


			LinearRegressionModel linearModel= linear.fit(features);

			LinearRegressionTrainingSummary linearSummary =	linearModel.summary();

			Dataset<Row> predictions=linearSummary.predictions();
			predictions.show(10,false);

			Dataset<Row> residuals=linearSummary.residuals();
			residuals.show(10,false);

			double r2=linearSummary.r2();
			System.out.println("R2 Measure :: "+r2);

			linearModel.write().overwrite().save(args[2]+"/linearModel");



			System.out.println("============================LinearRegressionModel End============================\n\n");

			System.out.println("============================LogisticRegression Start============================\n");


			// Create a LogisticRegression instance.
			LogisticRegression lr = new LogisticRegression();


			// We may set parameters using setter methods.
			lr.setMaxIter(200)
			.setRegParam(0.3)
			.setThreshold(0.1)
			.setElasticNetParam(0.2);

			// Learn a LogisticRegression model. 
			LogisticRegressionModel logisticmodel = lr.fit(features);

			LogisticRegressionTrainingSummary LogisticSummary =	logisticmodel.summary();

			Dataset<Row> logisticPredictions=LogisticSummary.predictions();
			logisticPredictions.show(10,false);

			double[] F1Measure=LogisticSummary.fMeasureByLabel();

			for(double f :F1Measure) {
				System.out.println("F1 Measure :: "+f);
			}

			double fMeasure = LogisticSummary.weightedFMeasure();
			System.out.println("\nF1 fMeasure :: "+fMeasure);

			logisticmodel.write().overwrite().save(args[2]+"/logisticmodel");



			System.out.println("============================LogisticRegression End============================\n");





			spark.stop();
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(0);
		}

	}
}
//./bin/spark-submit  --class org.njit.cloudcomputing.QualityTraining  --master "spark://Viveks-MacBook-Pro.local:7077"  /Users/vivekbaskar/eclipse-workspace/QualityTrainingEngine/target/QualityTraining-jar-with-dependencies.jar  /Users/vivekbaskar/eclipse-workspace/QualityTraining/TrainingDataset.csv spark://Viveks-MacBook-Pro.local:7077 Users/vivekbaskar/eclipse-workspace/QualityTraining


