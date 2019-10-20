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

    val activationLevels: List<Double>
        get() = mutableListOf<Double>().apply {
            val sum = neurons.sumByDouble { it.weight }
            for (ruleIndex in 0 until ruleCount) {
                add(neurons[ruleIndex].weight / sum)
            }
        }
}