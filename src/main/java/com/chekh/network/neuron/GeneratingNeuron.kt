package com.chekh.network.neuron

class GeneratingNeuron(inputCount: Int) {
    var generating: Double = 0.0
        private set
    var p: MutableList<Double> = MutableList(inputCount + 1) { 0.0 }
        private set

    fun calculateGenerating(x: List<Double>, weight: Double): Double {
        require(p.size - 1 == x.size)
        var sum = p[0]
        x.forEachIndexed { index, value -> sum += p[index + 1] * value }
        generating = weight * sum
        return generating
    }
}