package com.chekh.network.layer

import com.chekh.network.neuron.AggregationNeuron

class AggregationLayer(val ruleCount: Int) {
    var neurons: MutableList<AggregationNeuron> = mutableListOf()
        private set

    init {
        for (index in 0 until ruleCount) {
            neurons.add(AggregationNeuron())
        }
    }

    fun calculate(fuzzyLayer: FuzzyLayer) {
        require(ruleCount == fuzzyLayer.ruleCount)
        for (ruleIndex in 0 until ruleCount) {
            val muList = mutableListOf<Double>()
            for (fuzzyNeuronIndex in ruleIndex until fuzzyLayer.neurons.size step ruleCount) {
                muList.add(fuzzyLayer.neurons[fuzzyNeuronIndex].mu)
            }
            neurons[ruleIndex].calculateWeight(muList)
        }
    }

    fun activationLevel(ruleIndex: Int): Double {
        var sum = 0.0
        neurons.forEach { sum += it.weight }
        return neurons[ruleIndex].weight / sum
    }
}