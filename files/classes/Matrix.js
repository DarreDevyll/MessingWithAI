export class Matrix {
    constructor(rows, cols) {
      this.rows = rows;
      this.cols = cols;
      this.data = Array(rows).fill().map(() => Array(cols).fill(0));
    }

    // Ensure that the passed arguments are instances of Matrix class
    static ensureInstance(inputMatrix) {
        if (!(inputMatrix instanceof Matrix))
            throw new Error('Error calling Matrix method: Must be instance of Matrix');
    }

    static ensureInstance(inputMatrixA, inputMatrixB) {
        if (!(inputMatrixA instanceof Matrix))
            throw new Error('Error calling Matrix method: Must be instance of Matrix');
        if (!(inputMatrixB instanceof Matrix))
            throw new Error('Error calling Matrix method: Must be instance of Matrix');
    }

    static ensureAdditionDimenstions(inputMatrixA, inputMatrixB) {
        if (inputMatrixA.rows != inputMatrixB.rows || inputMatrixA.cols != inputMatrixB.cols)
            throw new Error('Error adding/subtracting matrices: Must be same dimensions');
    }

    static ensureMultiplicationDimensions(inputMatrixA, inputMatrixB) {
        if(inputMatrixA.rows == inputMatrixB.rows && inputMatrixA.cols == inputMatrixB.cols)
            return true;
        if (inputMatrixA.cols != inputMatrixB.rows)
            throw new Error('Error multiplying matrices: Cols of A must equal Rows of B');
    }

    // Static methods of Matrix class
    static randomize(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);

        for (let row = 0; row < inputMatrix.rows; row++)
            for (let col = 0; col < inputMatrix.cols; col++)
                inputMatrix.data[row][col] = Math.random() * 2 - 1;

        return inputMatrix;
    }

    static fromArray(inputArray) {
        let outputMatrix = new Matrix(inputArray.length, 1);
        for (let i = 0; i < inputArray.length; i++)
            outputMatrix.data[i][0] = inputArray[i];

        return outputMatrix;
    }

    static toArray(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);

        let outputArray = [];

        for (let row = 0; row < a.row; row++)
            for (let col = 0; col < inputMatrix.col; col++)
                outputArray.push(inputMatrix.data[row][col]);
        
        return outputArray;
    }

    static map(inputMatrix, func) {
        //Matrix.ensureInstance(inputMatrix);

        for (let row = 0; row < inputMatrix.rows; row++)
            for (let col = 0; col < inputMatrix.cols; col++)
                inputMatrix.data[row][col] = func(inputMatrix.data[row][col])

        return inputMatrix;
    }

    static transpose(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);

        let outputMatrix = new Matrix(inputMatrix.cols, inputMatrix.rows);

        for (let row = 0; row < inputMatrix.rows; row++)
            for (let col = 0; col < inputMatrix.cols; col++)
                outputMatrix.data[col][row] = inputMatrix.data[row][col];

        return outputMatrix;
    }

    static add(inputMatrixA, inputMatrixB) {
        //Matrix.ensureInstance(inputMatrixA, inputMatrixB);
        Matrix.ensureAdditionDimenstions(inputMatrixA, inputMatrixB);

        let outputMatrix = new Matrix(inputMatrixA.rows, inputMatrixA.cols);

        for (let row = 0; row < inputMatrixA.rows; row++)
            for (let col = 0; col < inputMatrixA.cols; col++)
                outputMatrix.data[row][col] = inputMatrixA.data[row][col] + inputMatrixB.data[row][col];

        return outputMatrix;
    }

    static subtract (inputMatrixA, inputMatrixB) {
        //Matrix.ensureInstance(inputMatrixA, inputMatrixB);
        Matrix.ensureAdditionDimenstions(inputMatrixA, inputMatrixB);

        let outputMatrix = new Matrix(inputMatrixA.rows, inputMatrixA.cols);

        for (let row = 0; row < inputMatrixA.rows; row++)
            for (let col = 0; col < inputMatrixA.cols; col++)
                outputMatrix.data[row][col] = inputMatrixA.data[row][col] - inputMatrixB.data[row][col];

        return outputMatrix;
    }

    static multiply(inputMatrixA, inputMatrixB) {
        //Matrix.ensureInstance(inputMatrixA, inputMatrixB);
        Matrix.ensureMultiplicationDimensions(inputMatrixA, inputMatrixB);

        let outputMatrix = new Matrix(inputMatrixA.rows, inputMatrixB.cols);

        for (let row = 0; row < outputMatrix.rows; row++) {
            for (let col = 0; col < outputMatrix.cols; col++) {
            let sum = 0;

            for (let Acol = 0; Acol < inputMatrixA.cols; Acol++)
                sum += inputMatrixA.data[row][Acol] * inputMatrixB.data[Acol][col];

            outputMatrix.data[row][col] = sum;
            }
        }

        return outputMatrix;
    }

    static scalarMultiply(inputMatrix, scale) {
        //Matrix.ensureInstance(inputMatrix);

        let outputMatrix = new Matrix(inputMatrix.rows, inputMatrix.cols);

        for (let row = 0; row < inputMatrix.rows; row++)
            for (let col = 0; col < inputMatrix.cols; col++)
                outputMatrix.data[row][col] = inputMatrix.data[row][col] * scale;

        return outputMatrix;
    }


    // Non-static methods of Matrix class
    randomize() {
        for (let row = 0; row < this.rows; row++)
            for (let col = 0; col < this.cols; col++)
                this.data[row][col] = Math.random() - 0.5;
    }

    fromArray(arr) {
        if (this.rows != arr.length || this.cols != 1)
            throw new Error('Tried writing matrix from array: Matrix and array are not the same size');

        for (let i = 0; i < arr.length; i++)
            this.data[i][0] = arr[i];
    }

    toArray() {
        let outputArray = [];

        for (let row = 0; row < this.rows; row++)
            for (let col = 0; col < this.cols; col++)
                outputArray.push(this.data[row][col]);

        return outputArray;
    }

    map(func) {
        for (let row = 0; row < this.rows; row++)
            for (let col = 0; col < this.cols; col++)
                this.data[row][col] = func(this.data[row][col]);
    }

    // This will rewrite dimensions of current matrix
    transpose(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);

        this.rows = inputMatrix.cols;
        this.cols = inputMatrix.rows;

        for(let row = 0; row < this.rows; row++)
            for(let col = 0; col < this.cols; col++)
                this.data[row][col] = inputMatrix.data[col][row];
    }

    transpose() {
        let output = new Matrix(this.cols, this.rows);
        
        for(let row = 0; row < output.rows; row++)
            for(let col = 0; col < output.cols; col++)
                output.data[row][col] = this.data[col][row];
        
        return output;
    }

    add(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);
        Matrix.ensureAdditionDimenstions(this, inputMatrix);

        for (let row = 0; row < this.rows; row++)
            for (let col = 0; col < this.cols; col++)
                this.data[row][col] = this.data[row][col] + inputMatrix.data[row][col];
    }

    subtract(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);
        Matrix.ensureAdditionDimenstions(this, inputMatrix);

        for (let row = 0; row < this.rows; row++)
            for (let col = 0; col < this.cols; col++)
                this.data[row][col] = this.data[row][col] - inputMatrix.data[row][col];
    }

    multiply(inputMatrix) {
        //Matrix.ensureInstance(inputMatrix);
        if(Matrix.ensureMultiplicationDimensions(this, inputMatrix)) {
            for (let row = 0; row < this.rows; row++)
                for (let col = 0; col < this.cols; col++)
                    this.data[row][col] *= inputMatrix.data[row][col];
            return;
        }

        let outputMatrix = new Matrix(this.rows, inputMatrix.cols);

        for (let row = 0; row < outputMatrix.rows; row++) {
            for (let col = 0; col < outputMatrix.cols; col++) {
            let sum = 0;

            for (let thisCol = 0; thisCol < this.cols; thisCol++)
                sum += this.data[row][thisCol] * inputMatrix.data[thisCol][col];

            outputMatrix.data[row][col] = sum;
            }
        }

        return outputMatrix;
    }

    scalarMultiply(scale) {
        for (let row = 0; row < this.rows; row++)
            for (let col = 0; col < this.cols; col++)
                this.data[row][col] *= scale;
    }

}