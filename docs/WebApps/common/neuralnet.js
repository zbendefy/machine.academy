
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

        for (let i = 0; i < neuronCount; ++i) {
            let acc = 0;
            let weightVector = layer.weightMx[i];
            for (let j = 0; j < weightsPerNeuron; ++j) {
                acc += weightVector[j] * input[j];
            }
            acc += layer.biases[i];
            ret[i] = this._Sigmoid( acc );
        }
        return ret;
    }

    Calculate(input){
        if (input.length != this.jsonData.layers[0].weightMx[0].length)
            console.error("Invalid input!");

        let current = input;
        for(let layer of this.jsonData.layers){
            current = this._CalculateLayer(current, layer);
        }

        return current;
    }

    Mutate(amount){
        for(let layer of this.jsonData.layers) {
            let neuronCount = layer.weightMx.length;
            let weightsPerNeuron = layer.weightMx[0].length;

            for (let i = 0; i < neuronCount; ++i) {
                for (let j = 0; j < weightsPerNeuron; ++j) {
                    layer.weightMx[i][j] += (Math.random()*2-1) * amount;
                }
                layer.biases[i] += (Math.random()*2-1) * amount;
            }
        }
    }

    DeepCopy(){
        return new NeuralNetwork( JSON.parse( JSON.stringify( this.jsonData ) ) );
    }

    GetNetworkAsJSON(){
        return JSON.stringify(this.jsonData, null, 2);
    }

    GetName(){ return this.jsonData.name; }
    
    GetDescription(){ return this.jsonData.description; }
}