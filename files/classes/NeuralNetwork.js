import { Matrix } from "./Matrix";

class Layer {
    constructor(layerSize, toLayerSize, learningRate) {
        this.size = layerSize;
        this.toSize = toLayerSize;
        this.learn = learningRate;

        // set values
        this.nodes = new Matrix(this.size, 1);

        this.bias = new Matrix(this.size, 1);
        this.bias.randomize();

        this.weights = new Matrix(this.toSize, this.size);
        this.weights.randomize();
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    dsigmoid(x) {
        return x * (1 - x);
    }

    resetLayer() {
        this.bias.randomize();
        this.weights.randomize();
    }

    incrementSize() {
        this.size++;

        oldNodes = this.nodes;
        this.nodes = new Matrix(this.size, 1);

        for(let row = 0; row < oldNodes.rows; row++)
            for(let col = 0; col < oldNodes.cols; col++)
                this.nodes.data[row][col] = oldNodes.data[row][col];

        // delete old nodes?
    }

    incrementOutput() {
        this.toSize++;

        oldWeights = this.weights;
        this.weights = new Matrix(this.toSize, this.size);
        this.weights.randomize();

        for(let row = 0; row < oldNodes.rows; row++)
            for(let col = 0; col < oldNodes.cols; col++)
                this.nodes.data[row][col] = oldNodes.data[row][col];

        // delete old nodes?
    }

    decrementSize() {
        console.log("Sorry, We haven't figured out how to decrement sizes yet.\n");
    }
    
    decrementOutput() {
        console.log("Sorry, We haven't figured out how to decrement outputs yet.\n");
    }
}

class input extends Layer() {
    constructor(layerSize, toLayerSize, learningRate) {
        this.size = layerSize;
        this.toSize = toLayerSize;
        this.learn = learningRate;

        // set values
        this.nodes = new Matrix(this.size, 1);

        this.weights = new Matrix(this.toSize, this.size);
        this.weights.randomize();
    }

    feed(input_array) {
        this.nodes.fromArray(input_array);
        let outputArray = Matrix.toArray(Matrix.multiply(this.weights, this.nodes));

        return outputArray;
    }

    train(delta_array) {
        let deltas = Matrix.fromArray(delta_array);
        let deltaWeights = Matrix.multiply(deltas, this.nodes);
        this.weights.subtract(deltaWeights);
    }
}

class output extends Layer() {
    constructor(layerSize, learningRate) {
        this.size = layerSize;
        this.learn = learningRate;

        // set values
        this.nodes = new Matrix(this.size, 1);

        this.bias = new Matrix(this.size, 1);
        this.bias.randomize();
    }

    get(input_array) {
        this.nodes.fromArray(input_array);
        this.nodes.add(this.bias);
        this.nodes.map(this.sigmoid);

        return this.nodes.toArray();
    }

    train(target_array) {
        let target = Matrix.fromArray(target_array);

        let outputError = Matrix.subtract(target, this.nodes);

        let outputGradient = Matrix.map(this.nodes, this.dsigmoid);
        outputGradient.multiply(outputError);
        outputGradient.scalarMultiply(this.learn);

        this.bias.add(outputGradient);

        return outputGradient;
    }

}

class feedForward extends Layer() {
    feed(input_array) {
        this.nodes.fromArray(input_array);
        this.nodes.add(this.bias);
        this.nodes.map(this.sigmoid);

        let outputArray = Matrix.toArray(Matrix.multiply(this.weights, this.nodes));
        return outputArray;
    }

    train(delta_array) {
        let deltas = Matrix.fromArray(delta_array);
        let deltaWeights = Matrix.multiply(deltas, this.nodes);
        this.weights.add(deltaWeights);

        let hiddenErrors = Matrix.multiply(Matrix.transpose(this.weights_ho), outputErrors);

        let hiddenGradients = Matrix.map(this.nodes, this.dsigmoid);
        hiddenGradients.multiply(hiddenErrors);
        hiddenGradients.scalarMultiply(this.learn);
        
        this.bias.add(hiddenGradients);

        return hiddenGradients.toArray();
    }
}


class Network {
    // Expects 2D matrix
    constructor(layerSizeArray, layerTypeArray, learningRate) {
        if (layerSizeArray.length != layerTypeArray.length)
            throw new Error('ERROR CONSTRUCTING NETWORK: Number of layers must be consistent when constructing a network');

        this.learningRate = learningRate;
    }


}