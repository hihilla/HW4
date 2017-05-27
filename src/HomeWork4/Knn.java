package HomeWork4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Knn implements Classifier {

	public enum EditMode {
		None, Forwards, Backwards
	};
	private class Neighbor implements Comparable<Neighbor> {
		public Instance instance;
		public double distance;
		
		public Neighbor(Instance instance, double distance) {
			this.instance = instance;
			this.distance = distance;
		}
		
		@Override
		public int compareTo(Neighbor neighbor) {
			double diff = this.distance - neighbor.distance;
			if (diff < 0) {
				return -1;
			} else if (diff > 0) {
				return 1;
			}
			return 0;
		}
	};

	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
	private int m_k; // {1, 2, …, 20}
	private int m_p; // {infinity(0), 1, 2, 3}
	private String m_majority; // {"uniform", "weighted"}
	
	public Knn(int k, int p, String majority) {
		this.m_k = k;
		this.m_p = p;
		this.m_majority = majority;
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
	}
	/**
	 * receives an instance and classify it according to it's k nearest neighbors
	 */
	@Override
	public double classifyInstance(Instance instance) {
		// array list of the kNN of the given instance
		ArrayList<Neighbor> kNN = findNearestNeighbors(instance);
		
		if (m_majority == "uniform"){
			return getClassVoteResult(kNN);
		} else { // majority == weighted for sure
			return getWeightedClassVoteResult(kNN);
		}
	}

	private void editedForward(Instances instances) {
		Instances data = new Instances(instances);
		for (Instance x : instances) {
			data.remove(x);
			m_trainingInstances = data;
			if (classifyInstance(x) != x.classValue()) {
				// if x is classify correctly by instances = {x},
				// remove x from instances. else - return x to instances!
				data.add(x);
			}
		}
		m_trainingInstances = data;
	}

	private void editedBackward(Instances instances) {
		m_trainingInstances = new Instances(instances);
		Instances data = new Instances(instances);
		data.delete();
		for (Instance x : instances) {
			if (classifyInstance(x) == x.classValue()) {
				// if x is NOT classify correctly by instances = {x},
				// add x to instances. else - remove x from instances!
				data.add(x);
			}
		}
		m_trainingInstances = new Instances(data);
	}

	/**
	 * Store the training set in the m_trainingInstances without editing.
	 * 
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
	 * Calculate the average error on a given instances set. The average error
	 * is the total number of classification mistakes on the input instances set
	 * and divides that by the number of instances in the input set
	 * 
	 * @param instace
	 * @return Average error (double).
	 */
	public double calcAvgError(Instances instances) {
		int numOfInstances = instances.numInstances();
		int numOfMistakes = 0;
		Instance instance;

		// goes through each instances and check the rediction is
		// it's actual classification, otherwise counts to mistakes
		for (int i = 0; i < numOfInstances; i++) {
			instance = instances.instance(i);
			if (instance.classValue() != classifyInstance(instance)) {
				numOfMistakes++;
			}
		}

		// calculates the average error...
		double avgError = numOfMistakes / (double) numOfInstances;

		return avgError;
	}

	/**
	 * Calculate the Precision & Recall on a given instances set.
	 * 
	 * @param instances
	 * @return double array of size 2. First index for Precision and the second
	 *         for Recall.
	 */
	public double[] calcConfusion(Instances instances) {
		double truePositive = 0; // prediction positive and condition positive 
		double falsePositive = 0; // prediction positive and condition negative
		double falseNegative = 0; // prediction negative and condition positive
		
		// count population
		for (Instance instance : instances) {
			boolean conditionPositive = instance.classValue() == 0;
			boolean predictionPositive = classifyInstance(instance) == 0;
			if (predictionPositive && conditionPositive) {
				truePositive++;
			} else if (predictionPositive && !conditionPositive) {
				falsePositive++;
			} else if (!predictionPositive && conditionPositive) {
				falseNegative++;
			}
		}
		
		// calculate precision and recall
		double precision = truePositive / (truePositive + falsePositive);
		double recall = truePositive / (truePositive + falseNegative);
		double[] ret = {precision, recall};
		return ret;
	}

	/**
	 * Find the K nearest neighbors for the instance being classified.
	 * 
	 * @param instance
	 * @return K nearest neighbors and their distances
	 */
	private ArrayList<Neighbor> findNearestNeighbors(Instance instance) {
		int numOfTrainingInstances = m_trainingInstances.size();
		Neighbor[] neighbors = new Neighbor[numOfTrainingInstances];

		// array of all instances and their distance from the given instance
		for (int i = 0; i < numOfTrainingInstances; i++) {
			neighbors[i] = new Neighbor(m_trainingInstances.get(i), 
										distance(m_trainingInstances.get(i), instance));
		}
		// convert the array of neighbors to an array list (in order to compare)
		ArrayList<Neighbor> neighborsList = new ArrayList<Neighbor>(Arrays.asList(neighbors));

		// sorts the array list of neighbors
		Collections.sort(neighborsList);

		// takes only the first (smallest) k neighbors and puts in a new array
		// list
		ArrayList<Neighbor> kNN = new ArrayList<>(neighborsList.subList(0, m_k - 1));

		return kNN;
	}

	/**
	 * Calculate the majority class of the neighbors.
	 * 
	 * @param neighbors
	 *            - a set of K nearest neighbors
	 * @return the majority vote on the class of the neighbors
	 */
	private double getClassVoteResult(ArrayList<Neighbor> neighbors) {
		if (neighbors.isEmpty()) {
			return 0; // arbitrary return value for empty neighbors set
		}
		int[] countClassifications = new int[m_trainingInstances.classAttribute().numValues()];
		for (Neighbor inst : neighbors) {
			countClassifications[(int) inst.instance.classValue()]++;
		}
		if (countClassifications[0] > countClassifications[1]) {
			return 0;
		}
		return 1;
	}

	/**
	 * Calculate the weighted majority class of the neighbors. In this method
	 * the class vote is normalized by the distance from the instance being
	 * classified. Instead of giving one vote to every class, you give a vote of
	 * 1/(distance from instance)^2
	 * 
	 * @param neighbors
	 *            - set of K nearest neighbors (and perhaps their distances)
	 * @return the majority vote on the class of the neighbors, where each
	 *         neighbor's class is weighted by the neighbors distance from the
	 *         instance being classified.
	 */
	private double getWeightedClassVoteResult(ArrayList<Neighbor> neighbors) {
		if (neighbors.isEmpty()) {
			return 0; // arbitrary return value for empty neighbors set
		}
		double[] countClassifications = new double[m_trainingInstances.classAttribute().numValues()];
		for (Neighbor inst : neighbors) {
			double vote = 1.0 / Math.pow(inst.distance, 2);
			System.out.println(vote);
			countClassifications[(int) inst.instance.classValue()] += vote;
		}
		if (countClassifications[0] > countClassifications[1]) {
			return 0;
		}
		return 1;
	}

	/**
	 * Calculate distance between two instances
	 * 
	 * @param first
	 *            instance
	 * @param second
	 *            instance
	 * @return input instances’ distance according to the distance function that
	 *         your algorithm is configured to use.
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
	 * 
	 * @param first
	 *            instance
	 * @param second
	 *            instance
	 * @return the l-p distance between the two instances note: p can be a
	 *         variable of your class or you can set p some other way.
	 */
	private double lpDistance(Instance first, Instance second) {
		double distance = 0;
		int numAttributes = first.numAttributes() - 1;
		for (int i = 0; i < numAttributes; i++) {
			double tempCalc = Math.abs(first.value(i) - second.value(i));
			tempCalc = Math.pow(tempCalc, m_p);
			distance += tempCalc;
		}
		return Math.pow(distance, 1.0 / m_p);
	}

	/**
	 * Calculate l-infinity distance between two instances
	 * 
	 * @param first
	 *            instance
	 * @param second
	 *            instance
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
