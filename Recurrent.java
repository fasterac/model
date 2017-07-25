package model;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Recurrent {
	
	private class Node {
		
		private ArrayList<Double> weights;
		private double recurrent;
		private double bias;
		private ArrayList<Double> values;
		private ArrayList<Double> losses;
		
		public Node() { this(new ArrayList<Double>(), 0, 0); }
		
		public Node(ArrayList<Double> weights, double recurrent, double bias) {
			
			setWeights(weights);
			setRecurrent(recurrent);
			setBias(bias);
			
		}
		
		
		public double getWeight(int index) { return weights.get(index); }
		
		public void setWeight(int index, double value) { weights.set(index, value); }
		
		public void setWeights(ArrayList<Double> weights) { this.weights = weights; }
		
		public double getRecurrent() { return recurrent; }
		
		public void setRecurrent(double recurrent) { this.recurrent = recurrent; }
		
		public double getBias() { return bias; }
		
		public void setBias(double bias) { this.bias = bias; }
		
		public void addValue(double value) { values.add(value); }
		
		public double getValue(int index) { return values.get(index); }
		
		public void resetValues() { values = new ArrayList<Double>(); }
		
		public double getLoss(int index) { return losses.get(index); }
		
		public void pushLoss(double value) { losses.add(0, value); }
		
		public void resetLosses() { losses = new ArrayList<Double>(); }
		
	}
	
	// constructor
	public int epoch = 0;
	public double overallLoss = 0;
	
	private ArrayList<ArrayList<Node>> layers = new ArrayList<>();
	
	public Recurrent(ArrayList<Integer> nodesInEachLayer) {
		
		for (int i = 0; i < nodesInEachLayer.size(); i++) {
			
			ArrayList<Node> layer = new ArrayList<>();
			
			for (int j = 0; j < nodesInEachLayer.get(i); j++) {
				
				if (i == 0) layer.add(new Node());
				
				else {
					
					ArrayList<Double> weights = new ArrayList<>();
					
					for (int k = 0; k < nodesInEachLayer.get(i - 1); k++) weights.add(randomDouble());
					
					layer.add(new Node(weights, randomDouble(), randomDouble()));
					
				}
				
			}
			
			layers.add(layer);
			
		}
		
	}
	
	public Recurrent(File file) throws Exception {
		
		Scanner s = new Scanner(file);
		
		int numOfFormerNodes = 0;
		
		int numOfLayers = s.nextInt();
		
		for (int i = 0; i < numOfLayers; i++) {
			
			ArrayList<Node> layer = new ArrayList<>();
			
			int numOfNodes = s.nextInt();
			
			for (int j = 0; j < numOfNodes; j++) {
				
				if (i == 0) layer.add(new Node());
				
				else {
					
					ArrayList<Double> weights = new ArrayList<>();
					
					for (int k = 0; k < numOfFormerNodes; k++) weights.add(s.nextDouble());
					
					layer.add(new Node(weights, s.nextDouble(), s.nextDouble()));
					
				}
				
			}
			
			layers.add(layer);
			
			numOfFormerNodes = numOfNodes;
			
		}
		
		s.close();
		
	}
	
	// utility
	private double rate = -0.01;
	
	private Random r = new Random(3);
	
	private double randomDouble() { return (r.nextDouble() - 0.5) * 3; }
	
	private double activate(double x) { return Math.tanh(x); }
	
	private double activatePrime(double activatedx) { return 1 - Math.pow(activatedx, 2); }
	
	// function
	public void feedForward(ArrayList<ArrayList<Double>> inputsSequence) {
		
		for (int t = 0; t < inputsSequence.size(); t++) {
			
			for (int i = 0; i < layers.size(); i++) {
				
				ArrayList<Node> layer = layers.get(i);
				
				for (int j = 0; j < layer.size(); j++) {
					
					Node node = layer.get(j);
					
					if (t == 0) node.resetValues();
					
					if (i == 0) node.addValue(inputsSequence.get(t).get(j));
					
					else {
						
						ArrayList<Node> inputLayer = layers.get(i - 1);
						
						double sum = 0;
						
						for (int k = 0; k < inputLayer.size(); k++) sum += inputLayer.get(k).getValue(t) * node.getWeight(k);
						
						if (t > 0) sum += node.getValue(t - 1) * node.getRecurrent();
						
						sum += node.getBias();
						
						node.addValue(activate(sum));
						
					}
					
				}
				
			}
			
		}
		
	}
	
	public void backPropagate(ArrayList<ArrayList<Double>> outputsSequence) {
		
		// backPropagate
		for (int t = outputsSequence.size() - 1; t >= 0; t--) {
			
			for (int i = layers.size() - 1; i > 0; i--) {
				
				ArrayList<Node> layer = layers.get(i);
				
				for (int j = 0; j < layer.size(); j++) {
					
					Node node = layer.get(j);
					
					if (t == outputsSequence.size() - 1) node.resetLosses();
					
					double sum = 0;
					
					if (i == layers.size() - 1) sum += node.getValue(t) - outputsSequence.get(t).get(j);
					
					else {
						
						ArrayList<Node> outputLayer = layers.get(i + 1);
						
						for (int k = 0; k < outputLayer.size(); k++) sum += outputLayer.get(k).getLoss(0) * outputLayer.get(k).getWeight(j);
						
					}
					
					if (t < outputsSequence.size() - 1) sum += node.getLoss(0) * node.getRecurrent();
					
					node.pushLoss(activatePrime(node.getValue(t)) * sum);
					
				}
				
			}
			
		}
		
		// adjustWeights
		for (int i = layers.size() - 1; i > 0; i--) {
			
			ArrayList<Node> layer = layers.get(i);
			
			for (int j = 0; j < layer.size(); j++) {
				
				Node node = layer.get(j);
				
				ArrayList<Node> inputLayer = layers.get(i - 1);
				
				for (int k = 0; k < inputLayer.size(); k++) {
					
					double deltaWeight = 0;
					
					for (int t = 0; t < outputsSequence.size(); t++) deltaWeight += inputLayer.get(k).getValue(t) * node.getLoss(t);
					
					node.setWeight(k, node.getWeight(k) + rate * deltaWeight);
					
				}
				
				double deltaRecurrent = 0;
				
				double deltaBias = 0;
				
				for (int t = 0; t < outputsSequence.size(); t++) {
					
					if (t > 0) deltaRecurrent += node.getValue(t - 1) * node.getLoss(t);
					
					deltaBias += node.getLoss(t);
					
				}
				
				node.setRecurrent(node.getRecurrent() + rate * deltaRecurrent);
				
				node.setBias(node.getBias() + rate * deltaBias);
				
			}
			
		}
		
		// updateInfo
		double loss = 0;
		
		ArrayList<Node> lastLayer = layers.get(layers.size() - 1);
		
		for (int i = 0; i < lastLayer.size(); i++) {
			
			Node node = lastLayer.get(i);
			
			double sum = 0;
			
			for (double nodeLoss : node.losses) sum += Math.abs(nodeLoss);
			
			loss += sum / node.losses.size();
			
		}
		
		overallLoss = (overallLoss * epoch + loss / lastLayer.size()) / (epoch + 1);
		
		epoch++;
		
	}
	
	public ArrayList<ArrayList<Double>> getResult() {
		
		ArrayList<ArrayList<Double>> result = new ArrayList<>();
		
		for (int i = 0; i < layers.get(layers.size() - 1).size(); i++) result.add(layers.get(layers.size() - 1).get(i).values);
		
		return result;
		
	}
	
}
