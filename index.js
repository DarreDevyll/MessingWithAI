import {RNN, FFN} from "./files/classes/oldNeuralNetwork.js";

// OLD NEURAL NETWORK TEST: RECURRENT AND FEED FORWARD NETWORK
//*
let recurrent = new RNN(3, 4, 2);
let feedForward = new FFN(3, 4, 5, 2);

// Provide some input data to the network
let input = [0.6660000000, 0.2950000000, 0.9820000000];

let recurrentOutput = recurrent.predict(input);
let feedForwardOutput = feedForward.feedforward(input);

// Initial Outputs
console.log('Initial Outputs: ');
console.log('Recurrent Output: ' + recurrentOutput);
console.log('Feed Forward Output: ' + feedForwardOutput + '\n');

// Retrain with value passed
let goal = [1.000000000, 0.000000000];

let accuracy = [[]];
let iteration = [];

for (let i = 0; i < 10000; i++) {
// Retraining Recurrent Neural Network
recurrent.train(input, goal);
recurrentOutput = recurrent.predict(input);

// Retraining Feed Forward Neural Network
feedForward.backprop(input, goal);
feedForwardOutput = feedForward.feedforward(input);

if (feedForwardOutput[0] !== feedForwardOutput[0])
{
    console.log('Iteration ' + i + ': ');
    feedForward.printNetwork(input);
    break;
}

}

// Initial Outputs
console.log('Retrained Outputs: ');
console.log('Recurrent Output: ' + recurrentOutput);
console.log('Feed Forward Output: ' + feedForwardOutput + '\n');//*/