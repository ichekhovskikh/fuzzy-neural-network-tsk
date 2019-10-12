package com.chekh.network.layer

import com.chekh.network.neuron.GeneratingNeuron

class GeneratingLayer(val inputCount: Int, val ruleCount: Int) {
    var neurons: MutableList<GeneratingNeuron> = mutableListOf()
        private set

    init {
        for (index in 0 until ruleCount) {
            neurons.add(GeneratingNeuron(inputCount))
        }
    }

    fun calculate(inputLayer: InputLayer, aggregationLayer: AggregationLayer) {
        require(inputCount == inputLayer.inputCount)
        require(ruleCount == aggregationLayer.ruleCount)
        aggregationLayer.neurons.forEachIndexed { aggregationIndex, aggregationNeuron ->
            neurons[aggregationIndex].calculateGenerating(inputLayer.x, aggregationNeuron.weight)
        }
    }

    fun asActivationMatrix(aggregationLayer: AggregationLayer): DoubleArray {
        require(ruleCount == aggregationLayer.ruleCount)
        val array = mutableListOf<Double>()
        neurons.forEachIndexed { ruleIndex, neuron ->
            neuron.p.forEach { _ -> array.add(aggregationLayer.activationLevel(ruleIndex)) }
        }
        return array.toDoubleArray()
    }

    fun setLinearParams(params: List<Double>) {
        require(params.size == ruleCount * (inputCount + 1))
        for (ruleIndex in 0 until neurons.size) {
            for (paramIndex in 0 until neurons[ruleIndex].p.size) {
                neurons[ruleIndex].p[paramIndex] = params[paramIndex + ruleIndex * (inputCount + 1)]
            }
        }
    }
}