package hw4;

import java.util.*;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

@SuppressWarnings("serial")
public class Knn extends Classifier {
	
	/**
	 * a helper neighbor class, wrapping an instance and its distance.
	 * 
	 * @author amir
	 *
	 */
	private class Neighbor implements Comparable<Neighbor> {
		public int id;
		public Instance instance;
		public double distance;
		
		@Override
		public int compareTo(Neighbor other) {
			return Double.compare(distance, other.distance);
		}
		
		@Override
		public String toString() {
			return "id: " + id + ", distance: " + distance + ", instance: " + instance.toString();
		}
	}

	private int k; 	// The k of kNN: k belongs to {1, ..., 30}
	private int p; 	// The p of l-p distance: p belongs to {0 (infinity), 1, 2, 3}
	private int v; 	// The voting function: 0 = plain, 1 = weighted
	
	private String M_MODE = "";
	private Instances m_trainingInstances;

	/**
	 * Constructor.
	 * 
	 * @param k
	 * @param p
	 * @param v
	 */
	public Knn(int k, int p, int v) {
		this.k = k;
		this.p = p;
		this.v = v;
	}
	
	/**
	 * Getter for M_MODE.
	 * 
	 * @return
	 */
	public String getM_MODE() {
		return M_MODE;
	}

	/**
	 * Setter for M_MODE.
	 * 
	 * @param m_MODE
	 */
	public void setM_MODE(String m_MODE) {
		M_MODE = m_MODE;
	}

	/**
	 * Builds the classifier.
	 * 
	 * @param arg0
	 */
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (M_MODE) {
			case "none":
				noEdit(arg0);
				break;
			case "forward":
				editedForward(arg0);
				break;
			case "backward":
				editedBackward(arg0);
				break;
			default:
				noEdit(arg0);
				break;
		}
	}
	
	/**
	 * Classify a given instance using kNN.
	 * 
	 * @param ins
	 * @return
	 */
	public double classify(Instance ins) {
		
		// Find its nearest neighbors
		List<Neighbor> neighbors = findNearestNeighbors(ins);
		
		// Take a vote according to the voting function
		if (v == 0) {
			return getClassVoteResult(neighbors);
		} else {
			return getWeightedClassVoteResult(neighbors);
		}
		
	}
	
	/**
	 * Return the k nearest neighbors to ins.
	 * 
	 * @param ins
	 * @return
	 */
	private List<Neighbor> findNearestNeighbors(Instance ins) {
		
		// Calculate the distance from all instances
		ArrayList<Neighbor> neighbors = new ArrayList<Neighbor>(); 
		for (int i=0; i<m_trainingInstances.numInstances(); i++) {
			Neighbor neighbor = new Neighbor();
			neighbor.id = i;
			neighbor.instance = m_trainingInstances.instance(i);
			neighbor.distance = distance(ins, neighbor.instance);
			neighbors.add(neighbor);
		}
		
		// Sort them by distance from ins
		Collections.sort(neighbors);
		
		// Return the k (or less) closest neighbors
		return neighbors.subList(0, Math.min(k, neighbors.size()));
		
	}
	
	/**
	 * Return the majority vote of all the neighbors.
	 * 
	 * @param neighbors
	 * @return
	 */
	private double getClassVoteResult(List<Neighbor> neighbors) {
		
		// Fill an array with the class value of all the neighbors
		double[] arr = new double[neighbors.size()];
		for (int i=0; i<neighbors.size(); i++) {
			arr[i] = neighbors.get(i).instance.classValue();
			
		}
		
		// Return the majority class value
		return findMajority(arr);
		
	}
	
	/**
	 * Return the weighted majority vote of all the neighbors.
	 * 
	 * @param neighbors
	 * @return
	 */
	private double getWeightedClassVoteResult(List<Neighbor> neighbors) {
		
		// Count the weighted votes for all possible class values 
		double[] arr = new double[m_trainingInstances.classAttribute().numValues()];
		for (int i=0; i<neighbors.size(); i++) {
			double classVal = neighbors.get(i).instance.classValue();
			double distance = neighbors.get(i).distance;
			if (distance != 0) {
				arr[(int) classVal] += 1.0/(distance*distance);
			} else {
				// If the distance to this neighbor is zero
				// we want to pick its class value
				return classVal;
			}
		}
		
		// The class value is the index with the largest value in the array
		int maxIndex = 0;
		for (int i=1; i<arr.length; i++) {
			if (arr[i] > arr[maxIndex]) {
				maxIndex = i;
			}
		}
		
		return (double) maxIndex;
		
	}
	
	/**
	 * Calculate the distance between the two instances according to the p member.
	 * 
	 * @param ins1
	 * @param ins2
	 * @return
	 */
	private double distance(Instance ins1, Instance ins2) {
		if (p == 0) {
			return lInfinityDistance(ins1, ins2);
		} else {
			return lPDistance(ins1, ins2);
		}
	}
	
	/**
	 * Calculate the l-p distance between the two instances.
	 * 
	 * @param ins1
	 * @param ins2
	 * @return
	 */
	private double lPDistance(Instance ins1, Instance ins2) {
		double sum = 0;
		for (int i=0; i<m_trainingInstances.numAttributes()-1; i++) {
			sum += Math.pow(Math.abs(ins1.value(i) - ins2.value(i)), p);
		}
		return Math.pow(sum, 1.0/p);
	}
	
	/**
	 * Calculate the l-infinity distance between the two instances.
	 * 
	 * @param ins1
	 * @param ins2
	 * @return
	 */
	private double lInfinityDistance(Instance ins1, Instance ins2) {
		double max = -1;
		for (int i=0; i<m_trainingInstances.numAttributes()-1; i++) {
			double tmp = Math.abs(ins1.value(i) - ins2.value(i));
			if (tmp > max) {
				max = tmp;
			}
		}
		return max;
	}
	
	/**
	 * Calculate the average error = #mistakes/#instances.
	 * 
	 * @param instances
	 * @return
	 */
	public double calcAvgError(Instances instances) {
		double count = 0;
		for (int i=0; i<instances.numInstances(); i++) {
			Instance ins = instances.instance(i);
			double realClassValue = ins.classValue();
			double predClassValue = classify(ins);
			if (realClassValue != predClassValue) {
				count++;
			}
		}
		return count / instances.numInstances();
	}
	
	private void editedForward(Instances instances) {

		// Create an empty instances set
		m_trainingInstances = new Instances(instances);
		m_trainingInstances.delete();

		// Add only instances that aren't classified correctly in the 'm_trainingInstances'
		for (int i=0; i<instances.numInstances(); i++) {
			Instance ins = instances.instance(i);
			if (ins.classValue() != classify(ins)) {
				m_trainingInstances.add(ins);
			}
		}
	}
	
	private void editedBackward(Instances instances) {

		m_trainingInstances = new Instances(instances);

		for (int i=m_trainingInstances.numInstances()-1; i>= 0; i--) {
			Instance x = m_trainingInstances.instance(i);

			// m_trainingInstances = T-{x}
			m_trainingInstances.delete(i);

			// If x is NOT classified correctly by T-{x}, we keep x.
			// (in other words, if x is classified correctly by T-{x}, remove x from T.)
			if (x.classValue() != classify(x)) {
				m_trainingInstances.add(x);
			}
		}
	}
	
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}
	
	/**
	 * Find the majority value of an array.
	 * 
	 * @param a
	 * @return
	 */
	private double findMajority(double[] a) {
		
	    if (a == null || a.length == 0) {
	        return 0;
	    }

	    Arrays.sort(a);

	    double previous = a[0];
	    double popular = a[0];
	    int count = 1;
	    int maxCount = 1;

	    for (int i=1; i<a.length; i++) {
	        if (a[i] == previous) {
	            count++;
	        } else {
	            if (count > maxCount) {
	                popular = a[i-1];
	                maxCount = count;
	            }
	            previous = a[i];
	            count = 1;
	        }
	    }
	    
	    return (count > maxCount) ? a[a.length-1] : popular;
	    
	}

}
