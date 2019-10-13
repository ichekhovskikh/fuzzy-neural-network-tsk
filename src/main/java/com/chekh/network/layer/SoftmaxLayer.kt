package com.chekh.network.layer

import com.chekh.network.neuron.SoftmaxNeuron

class SoftmaxLayer {
    private val neuron: SoftmaxNeuron = SoftmaxNeuron()
    val y: Double get() = neuron.y

    fun calculate(signal: Double, weightSum: Double) {
        neuron.calculateOutput(signal, weightSum)
    }
}