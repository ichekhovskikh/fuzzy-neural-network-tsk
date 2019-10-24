package com.chekh.network.layer

import com.chekh.network.dataset.Dataset
import com.chekh.network.neuron.FuzzyNeuron
import com.chekh.network.util.derivativeCenter
import com.chekh.network.util.derivativeSigma
import kotlin.math.abs

class FuzzyLayer(val inputCount: Int, val ruleCount: Int) {
    private val neurons: MutableList<FuzzyNeuron> = MutableList(inputCount * ruleCount) { FuzzyNeuron() }

    val muGroupedByRules: List<List<Double>>
        get() = mutableListOf<List<Double>>().apply {
            for (ruleIndex in 0 until ruleCount) {
                val ruleGroup = mutableListOf<Double>()
                for (inputIndex in 0 until inputCount) {
                    ruleGroup.add(getFuzzyNeuron(inputIndex, ruleIndex).mu)
                }
                add(ruleGroup)
            }
        }

    fun initNonLinearParams(dataset: Dataset) {
        initNonIntersectingNeuron(dataset)
        initIntersectNeuron()
    }

    private fun initNonIntersectingNeuron(dataset: Dataset) {
        for (ruleIndex in 0 until ruleCount step 2) {
            val cluster = dataset.getClass(ruleIndex / 2)
            for (inputIndex in 0 until inputCount) {
                val neuron = getFuzzyNeuron(inputIndex, ruleIndex)
                val paramsForOneInput = cluster.map { it.inputs[inputIndex] }
                neuron.center = paramsForOneInput.average()
            }
        }
    }

    private fun initIntersectNeuron() {
        for (ruleIndex in 1 until ruleCount step 2) {
            for (inputIndex in 0 until inputCount) {
                val firstNeuron = getFuzzyNeuron(inputIndex, ruleIndex - 1)
                val secondNeuron = getFuzzyNeuron(inputIndex, ruleIndex)
                val thirdNeuron = getFuzzyNeuron(inputIndex, ruleIndex + 1)
                secondNeuron.center = (firstNeuron.center + thirdNeuron.center) / 2.0
                val radius = abs(thirdNeuron.center - firstNeuron.center) / 2.0
                firstNeuron.sigma = radius
                secondNeuron.sigma = radius
                thirdNeuron.sigma = radius
            }
        }
    }

    fun calculate(inputs: List<Double>) {
        require(inputCount == inputs.size)
        inputs.forEachIndexed { inputIndex, input ->
            for (ruleIndex in 0 until ruleCount) {
                getFuzzyNeuron(inputIndex, ruleIndex).calculateMu(input)
            }
        }
    }

    fun correct(inputs: List<Double>, linearParamsGrouped: List<List<Double>>, error: Double, learningRate: Double) {
        val mu = muGroupedByRules
        inputs.forEachIndexed { inputIndex, input ->
            for (ruleIndex in 0 until ruleCount) {
                val neuron = getFuzzyNeuron(inputIndex, ruleIndex)
                var gradCenter = 0.0
                var gradSigma = 0.0
                linearParamsGrouped.forEachIndexed { groupIndex, linearParam ->
                    var sum = linearParam[0]
                    inputs.forEachIndexed { index, value -> sum += linearParam[index + 1] * value }
                    gradCenter += error * sum * neuron.derivativeCenter(ruleIndex, inputIndex, groupIndex, input, mu)
                    gradSigma += error * sum * neuron.derivativeSigma(ruleIndex, inputIndex, groupIndex, input, mu)
                }
                neuron.center = neuron.center - learningRate * gradCenter
                neuron.sigma = neuron.sigma - learningRate * gradSigma
            }
        }
    }

    private fun getFuzzyNeuron(inputIndex: Int, ruleIndex: Int): FuzzyNeuron {
        return neurons[ruleIndex + inputIndex * ruleCount]
    }
}