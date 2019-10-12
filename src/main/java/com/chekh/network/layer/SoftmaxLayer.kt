package com.chekh.network.layer

import com.chekh.network.neuron.SoftmaxNeuron

class SoftmaxLayer {
    var neuron: SoftmaxNeuron = SoftmaxNeuron()
        private set

    fun calculate(summingLayer: SummingLayer) {
        neuron.calculateOutput(summingLayer.summingNeuron.sum, summingLayer.weightNeuron.sum)
    }
}