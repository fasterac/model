package model;

import java.util.ArrayList;
import java.util.Random;

public class LongShortTermMemory {
	
	public class Node {
		
		private ArrayList<Double> weights;
		private double memory;
		private double bias;
		private ArrayList<Double> forgets;
		private ArrayList<Double> sigmoids;
		private ArrayList<Double> values;
		private ArrayList<Double> losses;
		
public Node() { this(new ArrayList<Double>(), 0, 0); }
		
		public Node(ArrayList<Double> weights, double memory, double bias) {
			
			setWeights(weights);
			setMemory(memory);
			setBias(bias);
			
		}
		
		
		public double getWeight(int index) { return weights.get(index); }
		
		public void setWeight(int index, double value) { weights.set(index, value); }
		
		public void setWeights(ArrayList<Double> weights) { this.weights = weights; }
		
		public double getMemory() { return memory; }
		
		public void setMemory(double memory) { this.memory = memory; }
		
		public double getBias() { return bias; }
		
		public void setBias(double bias) { this.bias = bias; }
		
		public void addForget(double forget) { forgets.add(forget); }
		
		public double getForget(int index) { return forgets.get(index); }
		
		public void addSigmoid(double sigmoid) { sigmoids.add(sigmoid); }
		
		public double getSigmoid(int index) { return sigmoids.get(index); }
		
		public void addValue(double value) { values.add(value); }
		
		public double getValue(int index) { return values.get(index); }
		
		public double getLoss(int index) { return losses.get(index); }
		
		public void pushLoss(double value) { losses.add(0, value); }
		
		public void reset() {
			
			forgets = new ArrayList<Double>();
			
			forgets.add(0.0);
			
			values = new ArrayList<Double>();
			
			sigmoids = new ArrayList<Double>();
			
			losses = new ArrayList<Double>();
			
		}
		
	}
	
	// constructor
	private ArrayList<ArrayList<Node>> layers = new ArrayList<>();
	
	private double rate = -0.01;

	public int epoch = 0;
	
	public double overallLoss = 0;
	
	// utility
	private Random r = new Random(3);
	
	private double randomDouble() { return (r.nextDouble() - 0.5) * 3; }
	
	private double tanh(double x) { return Math.tanh(x); }
	
	private double tanhPrime(double tanhOfx) { return 1 - Math.pow(tanhOfx, 2); }
	
	private double sigmoid(double x) { return 1 / (1 + Math.pow(Math.E, -x)); }
	
	private double sigmoidPrime(double sigmoidOfx) { return sigmoidOfx * (1 - sigmoidOfx); }
	
	// function
	public void feedForward(ArrayList<ArrayList<Double>> inputsSequence) {
		
		for (int t = 0; t < inputsSequence.size(); t++) {
			
			ArrayList<Double> inputs = inputsSequence.get(t);
			
			for (int i = 0; i < layers.size(); i++) {
				
				ArrayList<Node> layer = layers.get(i);
				
				for (int j = 0; j < layer.size(); j++) {
					
					Node node = layer.get(j);
					
					if (t == 0) node.reset();
					
					if (i == 0) node.addValue(inputs.get(j));
					
					else {
						
						ArrayList<Node> inputLayer = layers.get(i - 1);
						
						double value = 0;
						
						for (int k = 0; k < inputLayer.size(); k++) value += inputLayer.get(k).getValue(t) * node.getWeight(k);
						
						if (t > 0) value += node.getValue(t - 1) * node.getMemory();
						
						value += node.getBias();
						
						double sigmoid = sigmoid(value);
						
						node.addSigmoid(sigmoid);
						
						value = sigmoid * (value + node.getForget(t));
						
						node.addForget(value);
						
						node.addValue(tanh(value * sigmoid));
						
					}
					
				}
				
			}
			
		}
		
	}
	
	public void backPropagate(ArrayList<ArrayList<Double>> outputsSequence) {
		
		// งานยากละท่านเอ๋ย
		
	}
	
}
