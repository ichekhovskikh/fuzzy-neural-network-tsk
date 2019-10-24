package com.chekh.network.neuron

class GeneratingNeuron(inputCount: Int) {
    var generating: Double = 0.0
        private set
    var params: MutableList<Double> = MutableList(inputCount + 1) { 0.0 }
        private set

    fun calculateGenerating(inputs: List<Double>, weight: Double): Double {
        require(params.size - 1 == inputs.size)
        var sum = params[0]
        inputs.forEachIndexed { index, input -> sum += params[index + 1] * input }
        generating = weight * sum
        return generating
    }
}