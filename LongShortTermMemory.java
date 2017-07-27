package model;

import java.util.ArrayList;
import java.util.Random;

public class LongShortTermMemory {
	
	private class Node {
		
		private ArrayList<Double> weights;
		private double memory;
		private double bias;
		private double read;
		private double forget;
		private double write;
		private ArrayList<Double> tanhs;
		private ArrayList<Double> sigmoids;
		private ArrayList<Double> forgets;
		private ArrayList<Double> values;
		private ArrayList<Double> readLosses;
		private ArrayList<Double> forgetLosses;
		private ArrayList<Double> writeLosses;
		private ArrayList<Double> losses;
		
		public Node() { this(new ArrayList<Double>(), 0, 0); }
		
		public Node(ArrayList<Double> weights, double memory, double bias) {
			
			this.weights = weights;
			this.memory = memory;
			this.bias = bias;
			
		}
		
		private double getWeight(int index) { return weights.get(index); }
		
		private void setWeight(int index, double value) { weights.set(index, value); }
		
		private void addTanh(double tanh) { tanhs.add(tanh); }
		
		private double getTanh(int index) { return tanhs.get(index); }
		
		private void addSigmoid(double sigmoid) { sigmoids.add(sigmoid); }
		
		private double getSigmoid(int index) { return sigmoids.get(index); }
		
		private void addForget(double forget) { forgets.add(forget); }
		
		private double getForget(int index) { return forgets.get(index); }
		
		private void addValue(double value) { values.add(value); }
		
		private double getValue(int index) { return values.get(index); }
		
		private double getReadLoss(int index) { return readLosses.get(index); }
		
		private void pushReadLoss(double readLoss) { readLosses.add(0, readLoss); }
		
		private double getForgetLoss(int index) { return forgetLosses.get(index); }
		
		private void pushForgetLoss(double forgetLoss) { forgetLosses.add(0, forgetLoss); }
		
		private double getWriteLoss(int index) { return writeLosses.get(index); }
		
		private void pushWriteLoss(double writeLoss) { writeLosses.add(0, writeLoss); }
		
		private double getLoss(int index) { return losses.get(index); }
		
		private void pushLoss(double loss) { losses.add(0, loss); }
		
		private void resetFeed() {
			
			forgets = new ArrayList<Double>();
			
			tanhs = new ArrayList<Double>();
			
			sigmoids = new ArrayList<Double>();
			
			values = new ArrayList<Double>();
			
		}
		
		private void resetLoss() {
			
			readLosses = new ArrayList<Double>();
			
			forgetLosses = new ArrayList<Double>();
			
			writeLosses = new ArrayList<Double>();
			
			losses = new ArrayList<Double>();
			
		}
		
	}
	
	// constructor
	private ArrayList<ArrayList<Node>> layers = new ArrayList<>();
	
	private double rate = -0.01;

	public int epoch = 0;
	
	public double overallError = 0;
	
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
			
			// feedForward !! need polishing
			ArrayList<Double> inputs = inputsSequence.get(t);
			
			for (int i = 0; i < layers.size(); i++) {
				
				ArrayList<Node> layer = layers.get(i);
				
				for (int j = 0; j < layer.size(); j++) {
					
					Node node = layer.get(j);
					
					if (t == 0) node.resetFeed();
					
					if (i == 0) node.addValue(inputs.get(j));
					
					else {
						
						ArrayList<Node> inputLayer = layers.get(i - 1);
						
						double value = 0;
						
						for (int k = 0; k < inputLayer.size(); k++) value += inputLayer.get(k).getValue(t) * node.getWeight(k);
						
						if (t > 0) value += node.getValue(t - 1) * node.memory;
						
						value += node.bias;
						
						double sigmoid = sigmoid(value);
						
						node.addSigmoid(sigmoid);
						
						value = tanh(value);
						
						node.addTanh(value);
						
						value = value * sigmoid * node.read;
						
						if (t > 0) value = value + sigmoid * node.forget * node.getForget(t - 1);
						
						node.addForget(value);
						
						node.addValue(tanh(value * sigmoid * node.write));
						
					}
					
				}
				
			}
			
		}
		
	}
	
	public void backPropagate(ArrayList<ArrayList<Double>> outputsSequence) {
		
		// backPropagate
		for (int t = outputsSequence.size() - 1; t >= 0; t--) {
			
			ArrayList<Double> outputs = outputsSequence.get(t);
			
			for (int i = layers.size() - 1; i > 0; i--) {
				
				ArrayList<Node> layer = layers.get(i);
				
				for (int j = 0; j < layer.size(); j++) {
					
					Node node = layer.get(j);
					
					if (t == outputsSequence.size() - 1) node.resetLoss();
					
					double sigmoid = node.getSigmoid(t);
					
					double loss = 0;
					
					if (i == layers.size() - 1) loss = node.getValue(t) - outputs.get(j);
					
					else {
						
						ArrayList<Node> outputLayer = layers.get(i + 1);
						
						for (int k = 0; k < outputLayer.size(); k++) loss += outputLayer.get(k).getLoss(t);
						
					}
					
					loss = tanhPrime(node.getValue(t)) * loss;
					
					node.pushWriteLoss(node.getForget(t) * loss);
					
					loss = sigmoid * node.write * loss;
					
					if (t < outputsSequence.size() - 1) loss = loss + (node.getForgetLoss(0) / node.getForget(t)) * node.getSigmoid(t + 1);
					
					node.pushForgetLoss(node.getForget(t - 1) * loss);
					
					node.pushReadLoss(node.getTanh(t) * loss);
					
					loss = sigmoid * node.read * loss;
					
					node.pushLoss(loss);
					
				}
				
			}
			
		}
		
		// adjustWeights
		for (int i = layers.size() - 1; i > 0; i++) {
			
			ArrayList<Node> layer = layers.get(i);
			
			for (int j = 0; j < layer.size(); j++) {
				
				Node node = layer.get(j);
				
				ArrayList<Node> inputLayer = layers.get(i - 1);
				
				for (int k = 0; k < inputLayer.size(); k++) {
					
					double deltaWeight = 0;
					
					for (int t = 0; t < outputsSequence.size(); t++) deltaWeight += inputLayer.get(k).getValue(t) * node.getLoss(t);
					
					node.setWeight(k, node.getWeight(k) + rate * deltaWeight);
					
				}
				
				double deltaMemory = 0;
				
				double deltaBias = 0;
				
				double deltaRead = 0;
				
				double deltaForget = 0;
				
				double deltaWrite = 0;
				
				for (int t = 0; t < outputsSequence.size(); t++) {
					
					if (t > 0) deltaMemory += node.getValue(t - 1) * node.getLoss(t);
					
					deltaBias += node.getLoss(t);
					
					deltaRead += node.getReadLoss(t);

					deltaForget += node.getForgetLoss(t);

					deltaWrite += node.getWriteLoss(t);
					
				}
				
				node.memory += rate * deltaMemory;
				
				node.bias += rate * deltaBias;
				
				node.read += rate * deltaRead;
				
				node.forget += rate * deltaForget;
				
				node.write += rate * deltaWrite;
				
			}
			
		}
		
		// updateInfo
		double error = 0;
		
		ArrayList<Node> lastLayer = layers.get(layers.size() - 1);
		
		for (int i = 0; i < lastLayer.size(); i++) {
			
			Node node = lastLayer.get(i);
			
			double sum = 0;
			
			for (int t = 0; t < outputsSequence.size(); t++) sum += Math.abs(node.getValue(t) - outputsSequence.get(t).get(i));
			
			error += sum / outputsSequence.size();
			
		}
		
		overallError = (overallError * epoch + error / lastLayer.size()) / (epoch + 1);
		
		epoch++;
		
	}
	
}
