package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import com.oracle.webservices.internal.api.databinding.Databinding.Builder;

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
			int k, int p, String majority,
			EditMode editMode) {
		int numInstancesInFold = (int) (instances.numInstances() / (double) numFolds);
		double error = 0;
		// shuffling instances
		instances.randomize(new Random());

		// generating numFolds knn trees, calculating their error
		// and calculating the average error
		for (int i = 0; i < numFolds; i++) {
			Instances testData = new Instances(instances, instances.numInstances());
			Instances trainData = new Instances(instances, instances.numInstances());
			for (int j = 0; j < instances.numInstances(); j++) {
				int startIndex = i * numInstancesInFold;
				int endIndex = startIndex + numInstancesInFold;
				if ((j < startIndex) || (j > endIndex)) {
					// index in test range - insert to test data
					trainData.add(instances.instance(j));
				} else {
					// index not in test range - insert to train data
					testData.add(instances.instance(j));
				}
			}
			
			Knn thisKnn = new Knn(k, p, majority);
			thisKnn.setEditMode(editMode);
			try {
				thisKnn.buildClassifier(trainData);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// adding this testing data's error to the accumulator
			error += thisKnn.calcAvgError(testData);
		}

		return error / (double) numFolds;
	}

	private static void findHyperParamsForData(Instances cancerInstances, 
												Instances glassInstances) {
		HyperParameters glassParams = findHyperParams(glassInstances, 
														10, 
														EditMode.None);
		System.out.printf(
				"Cross validation error with K = %d, p = %d, "
				+ "majority function = %s for glass data is: %.7f\n",
				glassParams.k, glassParams.p, glassParams.majority, 
				glassParams.error);

		HyperParameters cancerParams = findHyperParams(cancerInstances, 
														10, 
														EditMode.None);
		System.out.printf(
				"Cross validation error with K = %d, p = %s, "
				+ "majority function = %s, for cancer data is: %.7f\n",
				cancerParams.k, cancerParams.p, cancerParams.majority, 
				cancerParams.error);

		Knn thisKnn = new Knn(cancerParams.k, 
								cancerParams.p, 
								cancerParams.majority);
		try {
			thisKnn.buildClassifier(cancerInstances);
		} catch (Exception e) {
			e.printStackTrace();
		}
		double[] confusion = thisKnn.calcConfusion(cancerInstances);

		System.out.printf("The average Precision for the cancer dataset is: %.7f \n"
				+ "The average Recall for the cancer dataset is: %.7f\n", confusion[0], confusion[1]);
		
		int[] numOfFolds = {glassInstances.numInstances(), 50, 10, 5, 3 };
		// for every possible number of folding, prints the relevant outputs
		for (int fold : numOfFolds) {
			System.out.println("----------------------------");
			System.out.printf("Results for %d folds:\n", fold);
			System.out.println("----------------------------");
			printGlassResult(glassInstances, glassParams, fold, EditMode.None);
			printGlassResult(glassInstances, glassParams, fold, EditMode.Forwards);
			printGlassResult(glassInstances, glassParams, fold, EditMode.Backwards);
		}
	}

	/**
	 * @param glassInstances
	 * @param glassParams
	 * @param fold
	 */
	private static void printGlassResult(Instances glassInstances, 
			HyperParameters glassParams, int fold, EditMode eMode) {
		long startTime = System.nanoTime();
		// calculating cross validation error
		double crossValError = crossValidationError(glassInstances, fold, 
				glassParams.k, glassParams.p, glassParams.majority, 
				eMode);
		// calculating total and average elapsed time
		long totalTime = System.nanoTime() - startTime;
		long avgTime = totalTime / fold;
		Knn knn = new Knn(glassParams.k, glassParams.p, glassParams.majority);
		knn.setEditMode(eMode);
		try {
			knn.buildClassifier(glassInstances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// calculate number of instances in training set of each fold
		int numInstancesInFold = (int) (knn.getNumInstances() / (double) fold) * (fold - 1);
		System.out.printf("Cross validation error of %s-Edited knn on glass dataset is %.7f\n", eMode, crossValError);
		System.out.print("and the average elapsed time is ");
		System.out.println(avgTime);
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

		for (int k = 1; k <= 20; k++) {
			int tempK = k;
			for (int p = 0; p < 4; p++) {
				int tempP = p;
				String[] majOpts = { "uniform", "weighted" };
				for (String tempMajority : majOpts) {
					double tempErr = crossValidationError(data, numFolds, tempK, tempP, tempMajority, eMode);
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
		double error;
		int numInstances = 0;

		public HyperParameters(int k, int p, String majority, EditMode editMode, double error) {
			super();
			this.k = k;
			this.p = p;
			this.majority = majority;
			this.editMode = editMode;
			this.error = error;
		}
	}
}
