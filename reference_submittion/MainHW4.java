package hw4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

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
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main (String [] args) throws Exception{

		RunResults r1 = run("glass.txt", "none");
		RunResults r2 = run("cancer.txt", "none");
		RunResults r3 = run("glass.txt", "forward");
		RunResults r4 = run("glass.txt", "backward");

		System.out.println(r1.getQ6FormattedResults());
		System.out.println(r2.getQ6FormattedResults());
		System.out.println(r1.getQ8FormattedResults());
		System.out.println(r3.getQ8FormattedResults());
		System.out.println(r4.getQ8FormattedResults());
		
	}

	/**
	 * Runs the Knn and returns the best parameters k, p, v (v: 0=uniform, 1=weighted).
	 * @param data
	 * @param mode
	 * @return Cross validation error with K = <my_k>, p = <my_p>, vote function = <either weighted or uniform> for <data_file> data is: <my_error>
	 * @throws Exception
     */
	private static RunResults run(String data, String mode) throws Exception {

		RunResults results = new RunResults();
		int numOfFolds = 10;

		// Iterate over all possible combinations of k, p and v
		for (int k=1; k<=30; k++) {
			for (int p=0; p<=3; p++) {
				for (int v=0; v<=1; v++) {

					// Load the data and randomize it
					Instances allTheData = loadData(data);
					allTheData.randomize(new Random());

					// Calculate the cross validation error for this configuration
					double xve = crossValidationError(allTheData, numOfFolds, k, p, v, mode, results);
					System.out.printf("for k=%d p=%d v=%d the xve is %f\n", k, p, v, xve);
					
					// Save the lowest error parameters
					if (xve < results.xve) {
						results.k = k;
						results.p = p;
						results.v = v;
						results.xve = xve;
					}
					
				}
			}
		}
		
		System.out.println("*********************************");
		System.out.printf("%s: min XVE is %f for k=%d p=%d v=%d\n", data, results.xve, results.k, results.p, results.v);

		results.name = data.substring(0, data.indexOf('.')); // Take the file name without extensions
		results.mode = mode;

		return results;
	}
	
	/**
	 * Split the data in 'allTheData' into 'numOfFolds' folds, and calculate the
	 * cross validation error using the given 'k', 'p' and 'v' kNN parameters.
	 * 
	 * @param allTheData
	 * @param numOfFolds
	 * @param k
	 * @param p
	 * @param v
	 * @return
	 */
	private static double crossValidationError(Instances allTheData, int numOfFolds, int k, int p, int v, String mode, RunResults runResults) {
		
		double xverror = 0;
		
		for (int i=0; i<numOfFolds; i++) {
			
			// Calculate the start and end indexes of the testing data
			int foldSize = allTheData.numInstances() / numOfFolds;
			int startIndex = i * foldSize;
			int endIndex = startIndex + foldSize - 1;
			
			// Last round - grab the remainder too
			if (i == numOfFolds - 1) {
				endIndex = allTheData.numInstances() - 1;
			}
			
			// Divide the data into training and testing groups
			Instances trainingData = new Instances(allTheData, allTheData.numInstances());
			Instances testingData = new Instances(allTheData, allTheData.numInstances());
			for (int j=0; j<allTheData.numInstances(); j++) {
				if (j >= startIndex && j <= endIndex) {
					testingData.add(allTheData.instance(j));
				} else {
					trainingData.add(allTheData.instance(j));
				}
			}
			
			// Train the classifier
			Knn myKnn = new Knn(k, p, v);
			try {
				myKnn.setM_MODE(mode);
				myKnn.buildClassifier(trainingData);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			// Calculate the average error for this fold
			double startTime = System.nanoTime();
			xverror += myKnn.calcAvgError(testingData);
			runResults.duration += System.nanoTime() - startTime;
		}
		runResults.duration /= 10;
		
		return xverror / numOfFolds;
		
	}

	public static class RunResults {

		public String name, mode;
		public int k, p, v;
		public double xve, duration;
		public RunResults(){
			k = -1;
			p = -1;
			v = -1;
			xve = Double.POSITIVE_INFINITY;
		}

		public String getQ6FormattedResults() {
			return String.format("Cross validation error with k = %d, p = %d, vote function = %s for %s is: %f", k, p, v==0?"uniform":"weighted", name, xve);
		}

		public String getQ8FormattedResults() {
			return String.format("Cross validation error of %s-edited knn on %s dataset is %f and the average elapsed time is %f", mode, name, xve, duration);
		}
	}

}
