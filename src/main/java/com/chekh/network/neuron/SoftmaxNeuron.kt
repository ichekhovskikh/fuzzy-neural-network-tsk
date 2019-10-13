package com.chekh.network.neuron

class SoftmaxNeuron {
    var y: Double = 0.0
        private set

    fun calculateOutput(signal: Double, weightSum: Double): Double {
        y = signal / weightSum
        return y
    }
}