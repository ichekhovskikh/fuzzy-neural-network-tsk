package com.chekh.network.layer

import com.chekh.network.neuron.FuzzyNeuron

class FuzzyLayer(val inputCount: Int, val ruleCount: Int) {
    var neurons: MutableList<FuzzyNeuron> = mutableListOf()
        private set

    init {
        val uniqueNeurons = createUniqueNeurons(ruleCount)
        for (index in 0 until inputCount) {
            uniqueNeurons.forEach {
                neurons.add(FuzzyNeuron(it))
            }
        }
    }

    fun calculate(inputLayer: InputLayer) {
        require(inputCount == inputLayer.inputCount)
        inputLayer.x.forEachIndexed { inputIndex, value ->
            for (ruleIndex in 0 until ruleCount) {
                neurons[ruleIndex + inputIndex * inputCount].calculateMu(value)
            }
        }
    }

    fun hybridCorrect(inputLayer: InputLayer, errors: List<Double>, learningRate: Double) {
        TODO("not implemented")
    }

    private fun createUniqueNeurons(count: Int): List<FuzzyNeuron> {
        val uniqueNeurons = mutableListOf<FuzzyNeuron>()
        for (index in 0 until count) {
            uniqueNeurons.add(FuzzyNeuron())
        }
        return uniqueNeurons
    }
}