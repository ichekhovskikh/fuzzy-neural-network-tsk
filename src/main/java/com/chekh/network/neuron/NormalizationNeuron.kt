package com.chekh.network.neuron

class NormalizationNeuron {
    var output: Double = 0.0
        private set

    fun calculateOutput(signal: Double, weightSum: Double): Double {
        output = signal / weightSum
        return output
    }
}