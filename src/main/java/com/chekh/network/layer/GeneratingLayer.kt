package com.chekh.network.layer

import com.chekh.network.neuron.GeneratingNeuron

class GeneratingLayer(val inputCount: Int, val ruleCount: Int, val outputCount: Int) {
    var neurons: MutableList<GeneratingNeuron> = mutableListOf()
        private set

    init {
        for (index in 0 until ruleCount * outputCount) {
            neurons.add(GeneratingNeuron(inputCount))
        }
    }

    fun calculate(inputLayer: InputLayer, aggregationLayer: AggregationLayer) {
        require(inputCount == inputLayer.inputCount)
        require(ruleCount == aggregationLayer.ruleCount)
        aggregationLayer.neurons.forEachIndexed { aggregationIndex, aggregationNeuron ->
            for (outputIndex in 0 until outputCount) {
                neurons[aggregationIndex * outputCount].calculateGenerating(inputLayer.x, aggregationNeuron.weight)
            }
        }
    }
}