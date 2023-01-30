import { Matrix } from "./Matrix.js";

export class RNN {
  constructor(input_nodes, hidden_nodes, output_nodes) {
      this.input_nodes = input_nodes;
      this.hidden_nodes = hidden_nodes;
      this.output_nodes = output_nodes;

      this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
      this.weights_hh = new Matrix(this.hidden_nodes, this.hidden_nodes);
      this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
      this.weights_ih.randomize();
      this.weights_hh.randomize();
      this.weights_ho.randomize();

      this.bias_h = new Matrix(this.hidden_nodes, 1);
      this.bias_o = new Matrix(this.output_nodes, 1);
      this.bias_h.randomize();
      this.bias_o.randomize();

      this.hidden_state = new Matrix(this.hidden_nodes, 1);

      this.output = new Matrix(output_nodes, 1);
      this.learningRate = 0.1;
  }

  sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
  }

  dsigmoid(x) {
      return x * (1 - x);
  }

  predict(input_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);

    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(Matrix.multiply(this.weights_hh, this.hidden_state));
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(this.sigmoid);

    // Update the hidden state
    this.hidden_state = hidden;

    // Generating the output's output!
    this.output = Matrix.multiply(this.weights_ho, hidden);
    this.output.add(this.bias_o);
    this.output.map(this.sigmoid);
    return this.output.toArray();
  }

  train(input_array, target_array) {
    //let output = this.predict(input_array);

    // Calculate the error for the output layer
    let outputErrors = Matrix.subtract(Matrix.fromArray(target_array), this.output);
    // Calculate the error for the hidden layer
    let hiddenErrors = Matrix.multiply(Matrix.transpose(this.weights_ho), outputErrors);

    // Calculate the gradient for the output layer
    let outputGradients = Matrix.map(this.output, this.dsigmoid);
    outputGradients.multiply(outputErrors);
    outputGradients.scalarMultiply(this.learningRate);

    // Calculate the delta weights for the output layer
    let hiddenT = Matrix.transpose(this.hidden_state);
    let deltaW = Matrix.multiply(outputGradients, hiddenT);

    // Update the weights for the output layer
    this.weights_ho.add(deltaW);

    // Calculate the gradient for the hidden layer
    let hiddenGradients = Matrix.map(this.hidden_state, this.dsigmoid);
    hiddenGradients.multiply(hiddenErrors);
    hiddenGradients.scalarMultiply(this.learningRate);

    // Calculate the delta weights for the hidden layer
    let inputT = Matrix.transpose(Matrix.fromArray(input_array));
    deltaW = Matrix.multiply(hiddenGradients, inputT);

    // Update the weights for the hidden layer
    this.weights_ih.add(deltaW);

    // Update the bias for the hidden layer
    this.bias_h.add(hiddenGradients);

    // Update the bias for the output layer
    this.bias_o.add(outputGradients);
  }
}




// FEED FORWARD NEURAL NETWORK
export class FFN {

  constructor(input_nodes, hidden_nodes, hidden_layers, output_nodes) {

    // Setting sizes for input, hidden, and output layers
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.hidden_layers = hidden_layers;
    this.output_nodes = output_nodes;

    // Weights for input to hidden layer, and hidden to output layer
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    // My hybrid code

    if (this.hidden_layers > 1)
    {
      this.weights_hh = new Array(this.hidden_layers - 1);
      this.hidden = new Array(this.hidden_layers);
      this.hidden_bias = new Array(this.hidden_layers);

      for (let i = 0; i < this.hidden_layers - 1; i++)
      {
        this.weights_hh[i] = new Matrix(this.hidden_nodes, this.hidden_nodes);
        this.weights_hh[i].randomize();
        this.hidden_bias[i] = new Matrix(this.hidden_nodes, 1);
        this.hidden_bias[i].randomize();
        this.hidden[i] = new Matrix(this.hidden_nodes, 1);
      }

      this.hidden_bias[this.hidden_layers - 1] = new Matrix(this.hidden_nodes, 1);
      this.hidden_bias[this.hidden_layers - 1].randomize();
      this.hidden[this.hidden_layers - 1] = new Matrix(this.hidden_nodes, 1);
    }
    else 
    {
      this.bias_h = new Matrix(this.hidden_nodes, 1); // Bias for hidden layers
      this.bias_h.randomize();
      this.hidden = new Matrix(this.hidden_nodes, 1); // Hidden Layer Values
    }

    // Hidden and Output layer biases
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_o.randomize();

    // Input, Output, and Learning Rate
    this.input = new Matrix(this.input_nodes, 1);
    this.output = new Matrix(this.output_nodes, 1);
    this.max_gradient = 1.0;
    this.max_error = 1.0;
    this.learningRate = 0.05;
  }

  sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
  }

  dsigmoid(x) {
      //return -(Math.log(1/x - 1));
      let fx = 1 / (1 + Math.exp(-x));
      return fx * (1 - fx);
  }

  clipGradients(gradients, maxValue) {
    for (let row = 0; row < gradients.rows; row++)
    {
      for (let col = 0; col < gradients.cols; col++)
      {
        if (gradients.data[row][col] > maxValue)
          gradients.data[row][col] = maxValue;
        else if (gradients.data[row][col] < -maxValue)
          gradients.data[row][col] = -maxValue;
      }
    }
    return gradients;
  }

  clipErrors(errors, maxValue) {
    let sum = 0;

    for (let row = 0; row < errors.rows; row++)
      for (let col = 0; col < errors.cols; col++)
        sum += errors.data[row][col];

    let scale = maxValue / sum;

    if (scale < maxValue)
      errors.scalarMultiply(maxValue);

    return errors;
  }


  feedforward(input_array) {
    // Generating the Hidden Outputs
    this.input.fromArray(input_array);

    if (this.hidden_layers == 1)
    {
      this.hidden = Matrix.multiply(this.weights_ih, this.input);
      this.hidden.add(this.bias_h);
      this.hidden.map(this.sigmoid);
      this.output = Matrix.multiply(this.weights_ho, this.hidden);
    }
    else
    {
      this.hidden[0] = Matrix.multiply(this.weights_ih, this.input);
      this.hidden[0].add(this.hidden_bias[0]);
      this.hidden[0].map(this.sigmoid);

      for (let i = 1; i < this.hidden_layers; i++)
      {
        this.hidden[i] = Matrix.multiply(this.weights_hh[i-1], this.hidden[i-1]);
        this.hidden[i].add(this.hidden_bias[i]);
        this.hidden[i].map(this.sigmoid);
      }
      this.output = Matrix.multiply(this.weights_ho, this.hidden[this.hidden_layers - 1]);
    }

    // Generating the networks output!
    this.output.add(this.bias_o);
    this.output.map(this.sigmoid);
    return this.output.toArray();
  }

  backprop(input_array, target_array) {
    // Convert input and target arrays to matrices
    this.feedforward(input_array);
    let targets = Matrix.fromArray(target_array);

    // Calculate the error for the output layer
    let outputErrors = Matrix.subtract(targets, this.output);
    outputErrors = this.clipErrors(outputErrors, this.max_error);

    // Calculate the gradients for the output layer
    let outputGradients = Matrix.map(this.output, this.dsigmoid);
    outputGradients.multiply(outputErrors);
    outputGradients.scalarMultiply(this.learningRate);
    outputGradients = this.clipGradients(outputGradients, this.max_gradient);
    this.bias_o.add(outputGradients);

    // Calculate the error for the hidden layer
    if (this.hidden_layers == 1)
    {
    let hiddenErrors = Matrix.multiply(Matrix.transpose(this.weights_ho), outputErrors);

    // Calculate the gradients for the hidden layer
    let hiddenGradients = Matrix.map(this.hidden, this.dsigmoid);
    hiddenGradients.multiply(hiddenErrors);
    hiddenGradients.scalarMultiply(this.learningRate);

    // Calculate deltas
    let hiddenTranspose = Matrix.transpose(hiddenErrors);
    let weightHO_deltas = Matrix.multiply(outputGradients, hiddenTranspose);

    let inputsTranspose = Matrix.transpose(this.input);
    let weightIH_deltas = Matrix.multiply(hiddenGradients, inputsTranspose);

    // Adjust the weights
    this.weights_ho.add(weightHO_deltas);
    this.weights_ih.add(weightIH_deltas);

    // Adjust the bias
    this.bias_h.add(hiddenGradients);
    }
    else
    {
      let hiddenErrors = new Array(this.hidden_layers);
      hiddenErrors[this.hidden_layers - 1] = Matrix.multiply(Matrix.transpose(this.weights_ho), outputErrors);
      hiddenErrors[this.hidden_layers - 1] = this.clipErrors(hiddenErrors[this.hidden_layers - 1], this.max_error);


      let hiddenGradients =  new Array(this.hidden_layers);
      hiddenGradients[this.hidden_layers - 1] = Matrix.map(this.hidden[this.hidden_layers - 1], this.dsigmoid);
      hiddenGradients[this.hidden_layers - 1].multiply(hiddenErrors[this.hidden_layers - 1]);
      hiddenGradients[this.hidden_layers - 1].scalarMultiply(this.learningRate);
      hiddenGradients[this.hidden_layers - 1] = this.clipGradients(hiddenGradients[this.hidden_layers - 1], this.max_gradient);
      this.hidden_bias[this.hidden_layers - 1].add(hiddenGradients[this.hidden_layers - 1]);

      let hiddenTranspose = new Array(this.hidden_layers);
      hiddenTranspose[this.hidden_layers - 1] = Matrix.transpose(hiddenErrors[this.hidden_layers - 1]);

      let weightHO_deltas = Matrix.multiply(outputGradients, hiddenTranspose[this.hidden_layers - 1]);
      this.clipGradients(weightHO_deltas, this.max_gradient);
      this.weights_ho.add(weightHO_deltas);

      for (let i = this.hidden_layers - 2; i >= 0; i--)
      {
        //console.log(this.weights_hh[i], i);
        hiddenErrors[i] = Matrix.multiply(Matrix.transpose(this.weights_hh[i]), hiddenErrors[i+1]);
        hiddenErrors[i] = this.clipErrors(hiddenErrors[i], this.max_error);
        hiddenGradients[i] = Matrix.map(this.hidden[i], this.dsigmoid);
        hiddenGradients[i].multiply(hiddenErrors[i]);
        hiddenGradients[i].scalarMultiply(this.learningRate);
        hiddenGradients[i] = this.clipGradients(hiddenGradients[i], this.max_gradient);
        this.hidden_bias[i].add(hiddenGradients[i]);
        
        hiddenTranspose[i] = Matrix.transpose(hiddenErrors[i]);
      }

      let weightHH_deltas = new Array(this.hidden_layers - 1);

      for (let i = this.hidden_layers - 2; i >= 0; i--)
      {
        weightHH_deltas[i] = Matrix.multiply(hiddenGradients[i+1], hiddenTranspose[i]);
        weightHH_deltas[i] = this.clipGradients(weightHH_deltas[i], this.max_gradient);
        this.weights_hh[i].add(weightHH_deltas[i]);
      }

      let inputsTranspose = Matrix.transpose(this.input);
      let weightIH_deltas = Matrix.multiply(hiddenGradients[0], inputsTranspose);
      weightIH_deltas = this.clipGradients(weightIH_deltas, this.max_gradient);
      this.weights_ih.add(weightIH_deltas);
    }
  }

  printNetwork(input_array) {

    console.log('Input Layer: ' + input_array);
    console.log('I->H Weights: ' + this.weights_ih.data);

    if (this.hidden_layers == 1) {
      this.hidden = Matrix.multiply(this.weights_ih, this.input);
      this.hidden.add(this.bias_h);
      this.hidden.map(this.sigmoid);
      console.log('Hidden Biases: ' + this.bias_h.data);
      console.log('Hidden Layer: ' + this.hidden.data);

      this.output = Matrix.multiply(this.weights_ho, this.hidden);
    }
    else
    {
      this.hidden[0] = Matrix.multiply(this.weights_ih, this.input);
      this.hidden[0].add(this.hidden_bias[0]);
      this.hidden[0].map(this.sigmoid);
      console.log('Hidden Bias 1: ' + this.hidden_bias[0].data);
      console.log('Hidden Layer 1: ' + this.hidden[0].data);

      for (let i = 1; i < this.hidden_layers; i++)
      {
        this.hidden[i] = Matrix.multiply(this.weights_hh[i-1], this.hidden[i-1]);
        this.hidden[i].add(this.hidden_bias[i]);
        this.hidden[i].map(this.sigmoid);

        console.log('H' + i + '->H' + (i+1) + ' Weights: ' + this.weights_hh[i - 1].data);
        console.log('Hidden Bias ' + (i+1) + ': ' + this.hidden_bias[i].data);
        console.log('Hidden Layer ' + (i+1) + ': ' + this.hidden[i].data);
      }
      this.output = Matrix.multiply(this.weights_ho, this.hidden[this.hidden_layers - 1]);
    }

    // Generating the networks output!
    this.output.add(this.bias_o);
    this.output.map(this.sigmoid);

    console.log('Output Bias: ' + this.bias_o.data);
    console.log('Final Output: ' + this.output.data);
  }
}

// BELOW IS CODE I HAVE YET TO MESS WITH
class MarkovChain {
  constructor(numStates) {
    this.numStates = numStates;
    this.transitionMatrix = [];
    for (let i = 0; i < this.numStates; i++) {
      this.transitionMatrix[i] = [];
      for (let j = 0; j < this.numStates; j++) {
        this.transitionMatrix[i][j] = 0;
      }
    }
  }

  train(sequence) {
    for (let i = 0; i < sequence.length - 1; i++) {
      let currentState = sequence[i];
      let nextState = sequence[i + 1];
      this.transitionMatrix[currentState][nextState]++;
    }

    for (let i = 0; i < this.numStates; i++) {
      let sum = 0;
      for (let j = 0; j < this.numStates; j++) {
        sum += this.transitionMatrix[i][j];
      }
      for (let j = 0; j < this.numStates; j++) {
        this.transitionMatrix[i][j] /= sum;
      }
    }
  }

  generate(startState) {
    let currentState = startState;
    let sequence = [currentState];
    while (true) {
      let nextState = this.nextState(currentState);
      if (nextState === undefined) {
        break;
      }
      sequence.push(nextState);
      currentState = nextState;
    }
    return sequence;
  }

  nextState(currentState) {
    let sum = 0;
    let r = Math.random();
    for (let i = 0; i < this.numStates; i++) {
      sum += this.transitionMatrix[currentState][i];
      if (sum >= r) {
        return i;
      }
    }
  }
}

class HopfieldNetwork {
  constructor(numNeurons) {
    this.numNeurons = numNeurons;
    this.weights = [];
    for (let i = 0; i < this.numNeurons; i++) {
      this.weights[i] = [];
      for (let j = 0; j < this.numNeurons; j++) {
        if (i === j) {
          this.weights[i][j] = 0;
        } else {
          this.weights[i][j] = Math.random() * 2 - 1;
        }
      }
    }
  }

  recall(input) {
    let output = [];
    for (let i = 0; i < this.numNeurons; i++) {
      let sum = 0;
      for (let j = 0; j < this.numNeurons; j++) {
        sum += input[j] * this.weights[i][j];
      }
      output[i] = sum >= 0 ? 1 : -1;
    }
    return output;
  }

  train(input) {
    for (let i = 0; i < this.numNeurons; i++) {
      for (let j = 0; j < this.numNeurons; j++) {
        if (i === j) continue;
        this.weights[i][j] += input[i] * input[j];
      }
    }
  }
}