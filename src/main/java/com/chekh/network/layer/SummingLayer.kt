package com.chekh.network.layer

import com.chekh.network.neuron.SummingNeuron

class SummingLayer {
    var summingNeuron = SummingNeuron()
    var weightNeuron = SummingNeuron()
        private set

    fun calculate(generatingLayer: GeneratingLayer, aggregationLayer: AggregationLayer) {
        require(aggregationLayer.ruleCount == generatingLayer.ruleCount)
        summingNeuron.calculateSum(generatingLayer.neurons.map { it.generating })
        weightNeuron.calculateSum(aggregationLayer.neurons.map { it.weight })
    }
}