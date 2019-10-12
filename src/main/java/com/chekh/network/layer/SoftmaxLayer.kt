package com.chekh.network.layer

import com.chekh.network.neuron.SoftmaxNeuron

class SoftmaxLayer {
    private val neuron: SoftmaxNeuron = SoftmaxNeuron()
    val y: Double get() = neuron.y

    fun calculate(signalFunction: Double, weightSum: Double) {
        neuron.calculateOutput(signalFunction, weightSum)
    }
}