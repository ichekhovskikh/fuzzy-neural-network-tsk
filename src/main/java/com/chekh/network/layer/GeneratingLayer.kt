package com.chekh.network.layer

import com.chekh.network.neuron.GeneratingNeuron
import java.util.concurrent.ThreadLocalRandom

class GeneratingLayer(val inputCount: Int, val ruleCount: Int) {
    private val neurons: MutableList<GeneratingNeuron> = MutableList(ruleCount) { GeneratingNeuron(inputCount) }

    val generating: List<Double> get() = neurons.map { it.generating }

    val paramsGroupedByRules: List<List<Double>>
        get() {
            val list = mutableListOf<List<Double>>()
            neurons.forEach { neuron -> list.add(neuron.params) }
            return list
        }

    fun initLinearParams() {
        val random = ThreadLocalRandom.current()
        neurons.forEach { neuron ->
            for (index in 0 until inputCount + 1) {
                neuron.params[index] = random.nextDouble(1.0)
            }
        }
    }

    fun calculate(inputs: List<Double>, weights: List<Double>) {
        require(inputCount == inputs.size)
        require(ruleCount == weights.size)
        weights.forEachIndexed { aggregationIndex, weight ->
            neurons[aggregationIndex].calculateGenerating(inputs, weight)
        }
    }

    fun getWeightedActivationLevels(inputs: List<Double>, activationLevels: List<Double>): DoubleArray {
        require(ruleCount == activationLevels.size)
        val array = mutableListOf<Double>()
        for (ruleIndex in 0 until neurons.size) {
            array.add(activationLevels[ruleIndex])
            inputs.forEach { input -> array.add(input * activationLevels[ruleIndex]) }
        }
        return array.toDoubleArray()
    }

    fun setLinearParams(params: List<Double>) {
        require(params.size == ruleCount * (inputCount + 1))
        for (ruleIndex in 0 until neurons.size) {
            for (paramIndex in 0 until neurons[ruleIndex].params.size) {
                neurons[ruleIndex].params[paramIndex] = params[paramIndex + ruleIndex * (inputCount + 1)]
            }
        }
    }
}