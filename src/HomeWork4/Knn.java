package HomeWork4;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Knn implements Classifier {
	
	public enum EditMode {None, Forwards, Backwards};
	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
	private int m_k; // {1, 2, …, 20}
	private int m_p; // {infinity(0), 1, 2, 3}
	private String m_majority; // {"uniform", "weighted"}
	
	private class Neighbor{
		public Instance instance;
		public double distance;
		
		public Neighbor(Instance instance, double distance){
			this.instance = instance;
			this.distance = distance;
		}
		
	}

	public EditMode getEditMode() {
		return m_editMode;
	}

	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}
	
	/**
	 * Builds a kNN from the training data. The method is already implemented 
	 * using switch statement on the enum EditMode. This enum set the edit mode 
	 * to one of its possibilities (None, Forwards, Backwards).
	 */
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (m_editMode) {
		case None:
			noEdit(arg0);
			break;
		case Forwards:
			editedForward(arg0);
			break;
		case Backwards:
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
		/* You should implement each one of the helper methods noEdit, 
		 * editedForward and editedBackward (the last 2 describe later). */
	}

	@Override
	public double classifyInstance(Instance instance) {
		//TODO: implement this method
		return 0;
	}

	private void editedForward(Instances instances) {
		//TODO: implement this method
	}

	private void editedBackward(Instances instances) {
		//TODO: implement this method
	}

	/**
	 * Store the training set in the m_trainingInstances without editing.
	 * @param instances
	 */
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * Calculate the average error on a given instances set. The average error is 
	 * the total number of classification mistakes on the input instances set and 
	 * divides that by the number of instances in the input set
	 * @param instace
	 * @return Average error (double). 
	 */
	private double calcAvgError(Instances instances) {
		int numOfInstances = instances.numInstances();
		int numOfMistakes= 0;
		Instance instance;
		
		// goes through each instances and check the rediction is
		// it's actual classification, otherwise counts to mistakes
		for (int i = 0; i < numOfInstances; i++){
			instance = instances.instance(i);
			if (instance.classValue() != classifyInstance(instance)){
				numOfMistakes++;
			}
		}
		
		// calculates the average error...
		double avgError = numOfMistakes / (double) numOfInstances; 
		
		return avgError;
	}
	
	/**
	 * Calculate the Precision & Recall on a given instances set.
	 * @param instances
	 * @return double array of size 2. First index for Precision and the second 
	 * for Recall.
	 */
	private double[] calcConfusion(Instances instances) {
		return null;
	}
	
	/**
	 * Calculate the cross validation error = average error on all folds.
	 * @param instances
	 * @return Average fold error (double)
	 */
	private double crossValidationError(Instances instances) {
		return 0;
	}
	
	/**
	 * Find the K nearest neighbors for the instance being classified.
	 * @param instance
	 * @return finds the K nearest neighbors (and perhaps their distances)
	 */
	private Instances findNearestNeighbors(Instance instance) {
		int numOfTrainingInstances = m_trainingInstances.size();
		int[] indexesOfkNN = 
		
		
		for (int i = 0; i < numOfTrainingInstances; i++){
			
		}
		
		
		return null;
	}
	
	/**
	 * Calculate the majority class of the neighbors.
	 * @param neighbors - a set of K nearest neighbors
	 * @return the majority vote on the class of the neighbors
	 */
	private double getClassVoteResult(Instances neighbors) {
		return 0;
	}
	
	/**
	 * Calculate the weighted majority class of the neighbors. In this method 
	 * the class vote is normalized by the distance from the instance being 
	 * classified. Instead of giving one vote to every class, you give a vote 
	 * of 1/(distance from instance)^2 
	 * @param neighbors - set of K nearest neighbors (and perhaps their distances)
	 * @return the majority vote on the class of the neighbors, where each 
	 * neighbor's class is weighted by the neighbor’s distance from the 
	 * instance being classified.
	 */
	private double getWeightedClassVoteResult(Instances neighbors) {
		return 0;
	}
	
	/**
	 * Calculate distance between two instances
	 * @param first instance
	 * @param second instance
	 * @return input instances’ distance according to the distance function that 
	 * your algorithm is configured to use.
	 */
	private double distance(Instance first, Instance second) {
		if (m_p == 0) {
			return lInfinityDistance(first, second);
		} else {
			return lpDistance(first, second);
		}
	}
	
	/**
	 * Calculate l-p distance between two instances
	 * @param first instance
	 * @param second instance
	 * @return the l-p distance between the two instances
	 * note: p can be a variable of your class or you can set p some other way.
	 */
	private double lpDistance(Instance first, Instance second) {
		double distance = 0;
		int numAttributes = first.numAttributes() - 1;
		for (int i = 0; i < numAttributes; i++) {
			double tempCalc = first.value(i) - second.value(i);
			tempCalc = Math.pow(tempCalc, m_p);
			distance += tempCalc;
		}
		return Math.pow(distance, 1.0 / m_p);
	}
	
	/**
	 * Calculate l-infinity distance between two instances
	 * @param first instance
	 * @param second instance
	 * @return the l-infinity distance between two instances
	 */
	private double lInfinityDistance(Instance first, Instance second) {
		double distance = Double.MIN_VALUE;
		int numAttributes = first.numAttributes() - 1;
		for (int i = 0; i < numAttributes; i++) {
			double tempCalc = Math.abs(first.value(i) - second.value(i));
			distance = Math.max(distance, tempCalc);
		}
		return distance;
	}
}
