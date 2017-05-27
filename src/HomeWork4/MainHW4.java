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
		// TODO: complete the Main method
	}

	/**
	 * Calculate the cross validation error = average error on all folds.
	 * 
	 * @param instances
	 * @return Average fold error (double)
	 */
	private double crossValidationError(Instances instances, int numFolds, int k, int p, String majority,
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

}
