package main.java.com.chekh.network.layer

import main.java.com.chekh.network.neuron.FuzzyNeuron

class FuzzyLayer(val inputCount: Int, val ruleCount: Int) {
    var neurons: MutableList<FuzzyNeuron> = mutableListOf()
        private set

    init {
        for (index in 0 until inputCount * ruleCount) {
            neurons.add(FuzzyNeuron())
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
}