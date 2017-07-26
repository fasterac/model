package model;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class FeedForward {
	
	private class Node {
		
		private ArrayList<Double> weights;
		private double bias;
		private double value;
		private double loss;
		
		public Node() { this(new ArrayList<Double>(), 0); }
		
		public Node(ArrayList<Double> weights, double bias) {
			
			setWeights(weights);
			setBias(bias);
			
		}
		
		
		public double getWeight(int index) { return weights.get(index); }
		
		public void setWeight(int index, double value) { weights.set(index, value); }
		
		public void setWeights(ArrayList<Double> weights) { this.weights = weights; }
		
		public double getBias() { return bias; }
		
		public void setBias(double bias) { this.bias = bias; }
		
		public double getValue() { return value; }
		
		public void setValue(double value) { this.value = value; }
		
		public double getLoss() { return loss; }
		
		public void setLoss(double loss) { this.loss = loss; }
		
	}
	
	// constructor
	private ArrayList<ArrayList<Node>> layers = new ArrayList<>();
	
	private double rate = -0.01;
	
	public int epoch = 0;
	
	public double overallLoss = 0;
	
	public FeedForward(ArrayList<Integer> nodesInEachLayer) {
		
		for (int i = 0; i < nodesInEachLayer.size(); i++) {
			
			ArrayList<Node> layer = new ArrayList<>();
			
			for (int j = 0; j < nodesInEachLayer.get(i); j++) {
				
				if (i == 0) layer.add(new Node());
				
				else {
					
					ArrayList<Double> weights = new ArrayList<>();
					
					for (int k = 0; k < nodesInEachLayer.get(i - 1); k++) weights.add(randomDouble());
					
					layer.add(new Node(weights, randomDouble()));
					
				}
				
			}
			
			layers.add(layer);
			
		}
		
	}
	
	public FeedForward(File file) throws Exception {
		
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
					
					layer.add(new Node(weights, s.nextDouble()));
					
				}
				
			}
			
			layers.add(layer);
			
			numOfFormerNodes = numOfNodes;
			
		}
		
		s.close();
		
	}
	
	// utility
	private Random r = new Random(3);
	
	private double randomDouble() { return (r.nextDouble() - 0.5) * 3; }
	
	private double tanh(double x) { return Math.tanh(x); }
	
	private double tanhPrime(double tanhOfx) { return 1 - Math.pow(tanhOfx, 2); }
	
	// function
	public void feedForward(ArrayList<Double> inputs) {
		
		for (int i = 0; i < layers.size(); i++) {
			
			ArrayList<Node> layer = layers.get(i);
			
			for (int j = 0; j < layer.size(); j++) {
				
				Node node = layer.get(j);
				
				if (i == 0) node.setValue(inputs.get(j));
				
				else {
					
					ArrayList<Node> inputLayer = layers.get(i - 1);
					
					double sum = 0;
					
					for (int k = 0; k < inputLayer.size(); k++) sum += inputLayer.get(k).getValue() * node.getWeight(k);
					
					sum += node.getBias();
					
					node.setValue(tanh(sum));
					
				}
				
			}
			
		}
		
	}
	
	public void backPropagate(ArrayList<Double> outputs) {
		
		// backPropagate
		int numOfLayers = layers.size();
		
		for (int i = numOfLayers - 1; i > 0; i--) {
			
			ArrayList<Node> layer = layers.get(i);
			
			for (int j = 0; j < layer.size(); j++) {
				
				Node node = layer.get(j);
				
				if (i == numOfLayers - 1) node.setLoss(tanhPrime(node.getValue()) * (node.getValue() - outputs.get(j)));
				
				else {
					
					ArrayList<Node> outputLayer = layers.get(i + 1);
					
					double sum = 0;
					
					for (int k = 0; k < outputLayer.size(); k++) sum += outputLayer.get(k).getLoss() * outputLayer.get(k).getWeight(j);
					
					node.setLoss(tanhPrime(node.getValue()) * sum);
					
				}
				
			}
			
		}
		
		// adjustWeights
		for (int i = numOfLayers - 1; i > 0; i--) {
			
			ArrayList<Node> layer = layers.get(i);
			
			ArrayList<Node> inputLayer = layers.get(i - 1);
			
			for (int j = 0; j < layer.size(); j++) {
				
				Node node = layer.get(j);
				
				for (int k = 0; k < inputLayer.size(); k++)  node.setWeight(k, node.getWeight(k) + rate * inputLayer.get(k).getValue() * node.getLoss());
				
				node.setBias(node.getBias() + rate * node.getLoss());
				
			}
			
		}
		
		// updateInfo
		double loss = 0;
		
		ArrayList<Node> lastLayer = layers.get(layers.size() - 1);
		
		for (int i = 0; i < lastLayer.size(); i++) loss += Math.abs(lastLayer.get(i).getLoss());
		
		overallLoss = (overallLoss * epoch + loss / lastLayer.size()) / (epoch + 1);
		
		epoch++;
		
	}
	
	public ArrayList<Double> getResult() {
		
		ArrayList<Double> result = new ArrayList<>();
		
		for (int i = 0; i < layers.get(layers.size() - 1).size(); i++) result.add(layers.get(layers.size() - 1).get(i).getValue());
		
		return result;
		
	}
	
}
