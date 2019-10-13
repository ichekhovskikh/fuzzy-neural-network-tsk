package com.chekh.network.layer

import com.chekh.network.neuron.GeneratingNeuron
import java.util.concurrent.ThreadLocalRandom

class GeneratingLayer(val inputCount: Int, val ruleCount: Int) {
    private val neurons: MutableList<GeneratingNeuron> = MutableList(ruleCount) { GeneratingNeuron(inputCount) }

    val generating: List<Double> get() = neurons.map { it.generating }

    val pGroupedByRules: List<List<Double>>
        get() {
            val list = mutableListOf<List<Double>>()
            neurons.forEach { neuron -> list.add(neuron.p) }
            return list
        }

    fun initWeights() {
        val random = ThreadLocalRandom.current()
        neurons.forEach { neuron ->
            var accumulator = 0.0
            for (index in 0 until inputCount) {
                val value = random.nextDouble(1.0 - accumulator)
                accumulator += value
                neuron.p[index] = value
            }
            neuron.p[inputCount] = 1.0 - accumulator
        }
    }

    fun calculate(x: List<Double>, weights: List<Double>) {
        require(inputCount == x.size)
        require(ruleCount == weights.size)
        weights.forEachIndexed { aggregationIndex, weight ->
            neurons[aggregationIndex].calculateGenerating(x, weight)
        }
    }

    fun asActivationArray(activationLevels: List<Double>): DoubleArray {
        require(ruleCount == activationLevels.size)
        val array = mutableListOf<Double>()
        neurons.forEachIndexed { ruleIndex, neuron ->
            neuron.p.forEach { _ -> array.add(activationLevels[ruleIndex]) }
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