package com.chekh.network.neuron

class GeneratingNeuron(inputCount: Int) {
    var generating: Double = 0.0
        private set
    var params: MutableList<Double> = MutableList(inputCount + 1) { 0.0 }
        private set

    fun calculateGenerating(x: List<Double>, weight: Double): Double {
        require(params.size - 1 == x.size)
        var sum = params[0]
        x.forEachIndexed { index, value -> sum += params[index + 1] * value }
        generating = weight * sum
        return generating
    }
}