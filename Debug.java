package model;

import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Debug {
	
	public static void main(String[] args) {
		
		Random r = new Random();
		
		Scanner s = new Scanner(System.in);
		
		ArrayList<Integer> list = new ArrayList<>();
		list.add(1); list.add(3); list.add(1);
		
		LongShortTermMemory lstm = new LongShortTermMemory(list);
		
		while (true) {
		
			for (int i = s.nextInt(); i > 0; i--) {
				
				ArrayList<ArrayList<Double>> inputsSequence = new ArrayList<>();
				ArrayList<ArrayList<Double>> outputsSequence = new ArrayList<>();
				
				int seqLength = 5;
				
				for (int j = 0; j < seqLength; j++) {
					
					ArrayList<Double> inputs = new ArrayList<>();
					ArrayList<Double> outputs = new ArrayList<>();
					
					inputs.add(j / 10.0);
					outputs.add(j / 10.0 + 0.1);
					
					inputsSequence.add(inputs);
					outputsSequence.add(outputs);
					
				}
				
				lstm.feedForward(inputsSequence);
				lstm.backPropagate(outputsSequence);
				
//				for (int j = 0; j < seqLength; j++) System.out.print(outputsSequence.get(j));
//				
//				System.out.println(" " + lstm.epoch + " " + lstm.overallError);
				
				ArrayList<ArrayList<Double>> result = lstm.getResult();
				
				for (int j = 0; j < result.size(); j++) System.out.print(result.get(j));

				System.out.println();
				
			}
			
			ArrayList<ArrayList<Double>> inputsSequence = new ArrayList<>();
			ArrayList<ArrayList<Double>> outputsSequence = new ArrayList<>();
			
			for (int j = 0; j < 5; j++) {
				
				ArrayList<Double> inputs = new ArrayList<>();
				ArrayList<Double> outputs = new ArrayList<>();
				
				inputs.add(j / 10.0);
				outputs.add(j / 10.0 + 0.1);
				
				inputsSequence.add(inputs);
				outputsSequence.add(outputs);
				
			}
			
			lstm.feedForward(inputsSequence);
			
			for (int j = 0; j < 5; j++) System.out.print(outputsSequence.get(j));
			
			System.out.println();
			
			ArrayList<ArrayList<Double>> result = lstm.getResult();
			
			for (int j = 0; j < result.size(); j++) System.out.print(result.get(j));
			
			System.out.println();
			
		}
		
	}

}
