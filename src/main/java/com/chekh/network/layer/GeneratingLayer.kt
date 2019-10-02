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

    fun asActivationMatrix(aggregationLayer: AggregationLayer): List<DoubleArray> {
        require(ruleCount == aggregationLayer.ruleCount)
        val matrix = mutableListOf<DoubleArray>()
        for (outputIndex in 0 until outputCount) {
            val array = mutableListOf<Double>()
            for ((ruleIndex, generatingIndex) in (outputIndex until neurons.size step outputCount).withIndex()) {
                neurons[generatingIndex].p.forEach { _ ->
                    array.add(aggregationLayer.activationLevel(ruleIndex))
                }
            }
            matrix.add(array.toDoubleArray())
        }
        return matrix
    }

    fun setLinearParams(params: List<Double>) {
        require(params.size == ruleCount * outputCount * (inputCount + 1))
        var index = 0
        for (outputIndex in 0 until outputCount) {
            for (generatingIndex in outputIndex until neurons.size step outputCount) {
                for (paramIndex in outputIndex until neurons[generatingIndex].p.size) {
                    neurons[generatingIndex].p[paramIndex] = params[index++]
                }
            }
        }
    }

    fun getErrors(inputLayer: InputLayer, errors: List<Double>): List<Double> {
        TODO("not implemented")
    }
}