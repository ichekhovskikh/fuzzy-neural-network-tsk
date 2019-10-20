package com.chekh.network.layer

import com.chekh.network.neuron.SoftmaxNeuron

class SoftmaxLayer {
    private val neuron: SoftmaxNeuron = SoftmaxNeuron()
    val output: Double get() = neuron.output

    fun calculate(signal: Double, weightSum: Double) {
        neuron.calculateOutput(signal, weightSum)
    }
}