package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import HomeWork4.Knn.EditMode;
import weka.core.Instances;

public class MainHW4 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		// load data
		Instances cancerData = loadData("/Users/hilla/Documents/B/ML/6/HW4/src/cancer.txt");
		Instances glassData = loadData("/Users/hilla/Documents/B/ML/6/HW4/src/glass.txt");

		// Finding the best hyper parameters using 10-folds cross validation,
		// for 2 different datasets ("glass" & "cancer")
		findHyperParamsForData(cancerData, glassData);
	}

	/**
	 * Calculate the cross validation error = average error on all folds.
	 * 
	 * @param instances
	 * @return Average fold error (double)
	 */
	private static double crossValidationError(Instances instances, int numFolds, 
			HyperParameters params) {
		int numInstancesInFold = (int) (instances.numInstances() / (double) numFolds);
		double error = 0;
		long totalTime = 0;

		// generating numFolds knn trees, calculating their error
		// and calculating the average error
		for (int i = 0; i < numFolds; i++) {
			Instances testData = new Instances(instances, instances.numInstances());
			Instances trainData = new Instances(instances, instances.numInstances());
			for (int j = 0; j < instances.numInstances(); j++) {
				if (j % numFolds == i) {
					testData.add(instances.instance(j));
				} else {
					trainData.add(instances.instance(j));
				}
			}

			Knn thisKnn = new Knn(params.k, params.p, params.majority);
			thisKnn.setEditMode(params.editMode);
			try {
				thisKnn.buildClassifier(trainData);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			params.numInstances += thisKnn.getNumInstances();
			
			// measuring time for this fold
			long startTime = System.nanoTime();
			// adding this testing data's error to the accumulator
			error += thisKnn.calcAvgError(testData);
			// adding this folds time to total time
			long endTime = System.nanoTime() - startTime;
			totalTime += endTime;
		}
		params.totalTime = totalTime;
		return error / (double) numFolds;
	}

	private static void findHyperParamsForData(Instances cancerInstances, Instances glassInstances) {
		HyperParameters glassParams = findHyperParams(glassInstances, 10, EditMode.None);
		System.out.printf("Cross validation error with K = %d, p = %d, " 
						+ "majority function = %s, for glass data is: %.7f\n",
						glassParams.k, glassParams.p, glassParams.majority, 
						glassParams.crossValidationError);

		HyperParameters cancerParams = findHyperParams(cancerInstances, 10, EditMode.None);
		System.out.printf("Cross validation error with K = %d, p = %s, " 
							+ "majority function = %s, for cancer data is: %.7f\n",
							cancerParams.k, cancerParams.p, cancerParams.majority, 
							cancerParams.crossValidationError);

		Knn thisKnn = new Knn(cancerParams.k, cancerParams.p, cancerParams.majority);
		try {
			thisKnn.buildClassifier(cancerInstances);
		} catch (Exception e) {
			e.printStackTrace();
		}
		double[] confusion = thisKnn.calcConfusion(cancerInstances);

		System.out.printf("The average Precision for the cancer dataset is: %.7f \n"
				+ "The average Recall for the cancer dataset is: %.7f\n", confusion[0], confusion[1]);

		int[] numOfFolds = { 3, 5, 10, 50, glassInstances.numInstances()};
//		 for every possible number of folding, prints the relevant outputs
		 for (int fold : numOfFolds) {
		 System.out.println("----------------------------");
		 System.out.printf("Results for %d folds:\n", fold);
		 System.out.println("----------------------------");
		 printGlassResult(glassInstances, glassParams, fold, EditMode.None);
		 printGlassResult(glassInstances, glassParams, fold,
		 EditMode.Forwards);
		 printGlassResult(glassInstances, glassParams, fold,
		 EditMode.Backwards);
		 }
	}

	/**
	 * @param glassInstances
	 * @param glassParams
	 * @param folds
	 */
	private static void printGlassResult(Instances glassInstances, HyperParameters glassParams, int folds,
			EditMode eMode) {
		// shuffling instances
		glassInstances.randomize(new Random());
		HyperParameters params = new HyperParameters(glassParams.k, glassParams.p, glassParams.majority, eMode, 0);
		// calculating cross validation error
		double crossValError = crossValidationError(glassInstances, folds, params);
		params.crossValidationError = crossValError;
		long totalTime = params.totalTime;
		double avgTime = totalTime / (double) folds;
		Knn knn = new Knn(params.k, params.p, params.majority);
		knn.setEditMode(params.editMode);
		try {
			knn.buildClassifier(glassInstances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// calculate number of instances in training set of each fold
		int numInstancesInFold = params.numInstances;
		System.out.printf("Cross validation error of %s-Edited knn on glass dataset is %.5f\n", 
				eMode, crossValError);
		System.out.printf("and the average elapsed time is %.5f\n", avgTime);
		System.out.print("The total elapsed time is: ");
		System.out.println(totalTime);
		System.out.print("The total number of instances used in the classification phase is: ");
		System.out.println(numInstancesInFold);
	}

	public static HyperParameters findHyperParams(Instances data, int numFolds, EditMode eMode) {
		int bestK = 0;
		int bestP = 0;
		String bestMajority = null;
		double bestErr = Double.MAX_VALUE;
		// shuffling instances
		data.randomize(new Random());

		for (int k = 1; k <= 20; k++) {
			int tempK = k;
			for (int p = 0; p < 4; p++) {
				int tempP = p;
				String[] majOpts = { "uniform", "weighted" };
				for (String tempMajority : majOpts) {
					HyperParameters params = new HyperParameters(tempK, tempP, tempMajority, eMode, 0);
					double tempErr = crossValidationError(data, numFolds, params);
					if (bestErr > tempErr) {
						bestK = tempK;
						bestP = tempP;
						bestMajority = tempMajority;
						bestErr = tempErr;
					}
				}
			}
		}
		return new HyperParameters(bestK, bestP, bestMajority, eMode, bestErr);
	}

	public static class HyperParameters {
		int k;
		int p;
		String majority;
		EditMode editMode;
		double crossValidationError;
		int numInstances = 0;
		long totalTime = 0;

		public HyperParameters(int k, int p, String majority, EditMode editMode, double error) {
			super();
			this.k = k;
			this.p = p;
			this.majority = majority;
			this.editMode = editMode;
			this.crossValidationError = error;
		}
	}
}
