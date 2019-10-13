package com.chekh.network.layer

import com.chekh.network.neuron.AggregationNeuron

class AggregationLayer(val ruleCount: Int) {
    private val neurons: MutableList<AggregationNeuron> = MutableList(ruleCount) { AggregationNeuron() }

    val weights: List<Double> get() = neurons.map { it.weight }

    fun calculate(muGrouped: List<List<Double>>) {
        muGrouped.forEachIndexed { ruleIndex, group ->
            neurons[ruleIndex].calculateWeight(group)
        }
    }

    fun asActivationArray(): List<Double> {
        val activationLevels = mutableListOf<Double>()
        val sum = neurons.sumByDouble { it.weight }
        for (ruleIndex in 0 until ruleCount) {
            activationLevels.add(neurons[ruleIndex].weight / sum)
        }
        return activationLevels
    }
}