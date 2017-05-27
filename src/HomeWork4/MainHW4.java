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
		
<<<<<<< HEAD
		// Finding the best hyper parameters using 10-folds cross validation,
=======
		// Finding the best hyper parameters using 10-folds cross validation, 
>>>>>>> f92835588da358232ab195233f5652cd7ab25fc4
		// for 2 different datasets ("glass" & "cancer")
		findHyperParams(cancerData, glassData);
	}

	/**
	 * Calculate the cross validation error = average error on all folds.
	 * 
	 * @param instances
	 * @return Average fold error (double)
	 */
	private static double crossValidationError(Instances instances, int numFolds, int k, int p, String majority,
			EditMode editMode) {
		int numInstancesInFold = (int) (instances.numInstances() / (double) numFolds);
		double error = 0;
		// shuffling instances
		instances.randomize(new Random());

		// generating numFolds knn trees, calculating their error
		// and calculating the average error
		for (int i = 0; i < numFolds; i++) {
			Instances testData = new Instances(instances, i, numInstancesInFold);
			Instances trainData = new Instances(instances);
			trainData.removeAll(testData);

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

	private static void findHyperParams(Instances cancerInstances, Instances glassInstances) {
		int bestK = 0;
		int bestP = 0;
		String bestMajority = null;
		double bestErr = Double.MAX_VALUE;
<<<<<<< HEAD

		for (int k = 1; k <= 20; k++) {
			int tempK = k;
			for (int p = 0; p < 4; p++) {
				int tempP = p;
				String[] majOpts = { "uniform", "weighted" };
				for (String tempMajority : majOpts) {
					double tempErr = crossValidationError(glassInstances, 10, tempK, tempP, tempMajority,
							EditMode.None);
					if (bestErr > tempErr) {
						bestK = tempK;
						bestP = tempP;
						bestMajority = tempMajority;
						bestErr = tempErr;
					}
				}
			}
		}
		System.out.printf("Cross validation error with K = %d,"
				+ " p = %d, majority function = %s for glass data is: %f\n",
				bestK, bestP, bestMajority, bestErr);
=======
//
//		for (int k = 1; k <= 20; k++) {
//			int tempK = k;
//			for (int p = 0; p < 4; p++) {
//				int tempP = p;
//				String[] majOpts = { "uniform", "weighted" };
//				for (String tempMajority : majOpts) {
//					double tempErr = crossValidationError(glassInstances, 10, 
//														tempK, tempP, tempMajority,
//														EditMode.None);
//					if (bestErr > tempErr) {
//						bestK = tempK;
//						bestP = tempP;
//						bestMajority = tempMajority;
//						bestErr = tempErr;
//					}
//				}
//			}
//		}
//		String pAsString = (bestP == 0)? "infinity" : Integer.toString(bestP);
//		System.out.printf("Cross validation error with K = %d,"
//				+ " p = %s, majority function = %s, for glass data is: %.5f\n",
//				bestK, pAsString, bestMajority, bestErr);
>>>>>>> f92835588da358232ab195233f5652cd7ab25fc4
		
		bestK = 0;
		bestP = 0;
		bestMajority = null;
		bestErr = Double.MAX_VALUE;
		
		for (int k = 1; k <= 20; k++) {
			int tempK = k;
			for (int p = 0; p < 4; p++) {
				int tempP = p;
				String[] majOpts = { "uniform", "weighted" };
				for (String tempMajority : majOpts) {
					double tempErr = crossValidationError(cancerInstances, 10, 
													tempK, tempP, tempMajority,
													EditMode.None);
					if (tempErr == 0) {
						System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
					}
					if (bestErr > tempErr) {
						bestK = tempK;
						bestP = tempP;
						bestMajority = tempMajority;
						bestErr = tempErr;
					}
				}
			}
		}
<<<<<<< HEAD
		System.out.printf("Cross validation error with K = %d, p = %d,"
				+ " majority function = %s for cancer data is: %f\n",
				bestK, bestP, bestMajority, bestErr);
=======
		String pAsString = (bestP == 0)? "infinity" : Integer.toString(bestP);
		System.out.printf("Cross validation error with K = %d, p = %s,"
				+ " majority function = %s, for cancer data is: %.5f\n",
				bestK, pAsString, bestMajority, bestErr);
>>>>>>> f92835588da358232ab195233f5652cd7ab25fc4
		
		Knn thisKnn = new Knn(bestK, bestP, bestMajority);
		try {
			thisKnn.buildClassifier(cancerInstances);
		} catch (Exception e) {
			e.printStackTrace();
		}
		double[] confusion = thisKnn.calcConfusion(cancerInstances);
		
		System.out.printf("The average Precision for the cancer dataset is: %f \n"
<<<<<<< HEAD
				+ " The average Recall for the cancer dataset is: %f\n",
=======
				+ "The average Recall for the cancer dataset is: %f\n",
>>>>>>> f92835588da358232ab195233f5652cd7ab25fc4
				confusion[0], confusion[1]);
	}
}
