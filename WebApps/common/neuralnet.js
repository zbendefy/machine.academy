
class NeuralNetwork
{
    constructor(jsonObject)
    {
        this.jsonData = jsonObject;
    }

    _Sigmoid(x){
        return 1.0 / (1.0 + Math.exp(-x));
    }

    _CalculateLayer(input, layer) {
        let ret = [];
        let neuronCount = layer.weightMx.length;
        let weightsPerNeuron = layer.weightMx[0].length;

        for (let i = 0; i < neuronCount; i++) {
            let acc = 0;
            for (let j = 0; j < weightsPerNeuron; j++) {
                acc += layer.weightMx[i][j] * input[j];
            }
            acc += layer.biases[i];
            ret[i] = this._Sigmoid( acc );
        }
        return ret;
    }

    Calculate(input){
        let current = input;
        for(let layer of this.jsonData.layers){
            current = this._CalculateLayer(current, layer);
        }

        return current;
    }

    GetName(){ return this.jsonData.name; }
    
    GetDescription(){ return this.jsonData.description; }
}