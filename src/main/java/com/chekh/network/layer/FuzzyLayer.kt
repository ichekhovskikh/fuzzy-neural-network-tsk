package com.chekh.network.layer

import com.chekh.network.Dataset
import com.chekh.network.neuron.FuzzyNeuron
import com.chekh.network.util.derivativeB
import com.chekh.network.util.derivativeC
import com.chekh.network.util.derivativeSigma
import kotlin.math.abs

class FuzzyLayer(val inputCount: Int, val ruleCount: Int) {
    private val neurons: MutableList<FuzzyNeuron> = MutableList(inputCount * ruleCount) { FuzzyNeuron() }

    val muGroupedByRules: List<List<Double>>
        get() {
            val list = mutableListOf<List<Double>>()
            for (ruleIndex in 0 until ruleCount) {
                val ruleGroup = mutableListOf<Double>()
                for (inputIndex in 0 until inputCount) {
                    ruleGroup.add(getFuzzyNeuron(inputIndex, ruleIndex).mu)
                }
                list.add(ruleGroup)
            }
            return list
        }

    fun initWeights(dataset: Dataset) {
        for (ruleIndex in 0 until ruleCount) {
            val cluster = dataset.rows.filter { it.output.toInt() == ruleIndex }
            for (inputIndex in 0 until inputCount) {
                val fuzzyNeuron = getFuzzyNeuron(inputIndex, ruleIndex)
                fuzzyNeuron.b = 1.0
                val paramsForOneInput = cluster.map { it.inputs[inputIndex] }
                fuzzyNeuron.c = paramsForOneInput.average()
                fuzzyNeuron.sigma = paramsForOneInput.maxBy { abs(it - fuzzyNeuron.c) }!!
            }
        }
    }

    fun calculate(x: List<Double>) {
        require(inputCount == x.size)
        x.forEachIndexed { inputIndex, value ->
            for (ruleIndex in 0 until ruleCount) {
                getFuzzyNeuron(inputIndex, ruleIndex).calculateMu(value)
            }
        }
    }

    fun correct(x: List<Double>, pGrouped: List<List<Double>>, error: Double, learningRate: Double) {
        val mu = muGroupedByRules
        x.forEachIndexed { inputIndex, input ->
            for (ruleIndex in 0 until ruleCount) {
                val fuzzyNeuron = getFuzzyNeuron(inputIndex, ruleIndex)
                var gradC = 0.0
                var gradSigma = 0.0
                var gradB = 0.0
                pGrouped.forEachIndexed { groupIndex, p ->
                    var sum = p[0]
                    x.forEachIndexed { index, value -> sum += p[index + 1] * value }
                    gradC += error * sum * fuzzyNeuron.derivativeC(ruleIndex, inputIndex, groupIndex, input, mu)
                    gradSigma += error * sum * fuzzyNeuron.derivativeSigma(ruleIndex, inputIndex, groupIndex, input, mu)
                    gradB += error * sum * fuzzyNeuron.derivativeB(ruleIndex, inputIndex, groupIndex, input, mu)
                }
                fuzzyNeuron.c = fuzzyNeuron.c - learningRate * gradC
                fuzzyNeuron.sigma = fuzzyNeuron.sigma - learningRate * gradSigma
                fuzzyNeuron.b = fuzzyNeuron.b - learningRate * gradB
            }
        }
    }

    private fun getFuzzyNeuron(inputIndex: Int, ruleIndex: Int): FuzzyNeuron {
        return neurons[ruleIndex + inputIndex * inputCount]
    }
}