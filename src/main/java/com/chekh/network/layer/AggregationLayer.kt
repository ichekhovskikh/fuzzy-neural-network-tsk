package com.chekh.network.layer

import com.chekh.network.neuron.AggregationNeuron

class AggregationLayer(val ruleCount: Int) {
    private val neurons: MutableList<AggregationNeuron> = mutableListOf()
    val weights: List<Double> get() = neurons.map { it.weight }

    init {
        for (index in 0 until ruleCount) {
            neurons.add(AggregationNeuron())
        }
    }

    fun calculate(muGrouped: List<List<Double>>) {
        muGrouped.forEachIndexed { ruleIndex, group ->
            neurons[ruleIndex].calculateWeight(group)
        }
    }

    fun asActivationArray(): List<Double> {
        val activationLevels = mutableListOf<Double>()
        for (ruleIndex in 0 until ruleCount) {
            var sum = 0.0
            neurons.forEach { sum += it.weight }
            activationLevels.add(neurons[ruleIndex].weight / sum)
        }
        return activationLevels
    }
}