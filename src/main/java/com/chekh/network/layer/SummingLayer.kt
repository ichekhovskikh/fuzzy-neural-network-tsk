package com.chekh.network.layer

import com.chekh.network.neuron.SummingNeuron

class SummingLayer(val outputCount: Int) {
    var functionalNeurons: MutableList<SummingNeuron> = mutableListOf()
    var weightNeuron = SummingNeuron()
        private set

    init {
        for (index in 0 until outputCount) {
            functionalNeurons.add(SummingNeuron())
        }
    }

    fun calculate(generatingLayer: GeneratingLayer, aggregationLayer: AggregationLayer) {
        require(outputCount == generatingLayer.outputCount)
        require(aggregationLayer.ruleCount == generatingLayer.ruleCount)
        for (functionalIndex in 0 until outputCount) {
            val values = mutableListOf<Double>()
            for (generatingNeuronIndex in functionalIndex until generatingLayer.neurons.size step outputCount) {
                values.add(generatingLayer.neurons[generatingNeuronIndex].generating)
            }
            functionalNeurons[functionalIndex].calculateSum(values)
        }
        weightNeuron.calculateSum(aggregationLayer.neurons.map { it.weight })
    }
}