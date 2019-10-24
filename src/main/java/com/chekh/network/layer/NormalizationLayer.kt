package com.chekh.network.layer

import com.chekh.network.neuron.NormalizationNeuron

class NormalizationLayer {
    private val neuron: NormalizationNeuron = NormalizationNeuron()
    val output: Double get() = neuron.output

    fun calculate(signal: Double, weightSum: Double) {
        neuron.calculateOutput(signal, weightSum)
    }
}