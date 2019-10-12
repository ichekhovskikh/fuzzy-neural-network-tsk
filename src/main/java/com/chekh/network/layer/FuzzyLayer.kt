package com.chekh.network.layer

import com.chekh.network.neuron.FuzzyNeuron
import com.chekh.network.util.derivativeB
import com.chekh.network.util.derivativeC
import com.chekh.network.util.derivativeSigma

class FuzzyLayer(val inputCount: Int, val ruleCount: Int) {
    private val neurons: MutableList<FuzzyNeuron> = mutableListOf()
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

    init {
        val uniqueNeurons = createUniqueNeurons(ruleCount)
        for (index in 0 until inputCount) {
            uniqueNeurons.forEach {
                neurons.add(FuzzyNeuron(it))
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
                    val derivativeC = fuzzyNeuron.derivativeC(ruleIndex, inputIndex, groupIndex, input, mu)
                    val derivativeSigma = fuzzyNeuron.derivativeSigma(ruleIndex, inputIndex, groupIndex, input, mu)
                    val derivativeB = fuzzyNeuron.derivativeB(ruleIndex, inputIndex, groupIndex, input, mu)
                    gradC += error * sum * derivativeC
                    gradSigma += error * sum * derivativeSigma
                    gradB += error * sum * derivativeB
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

    private fun createUniqueNeurons(count: Int): List<FuzzyNeuron> {
        val uniqueNeurons = mutableListOf<FuzzyNeuron>()
        for (index in 0 until count) {
            uniqueNeurons.add(FuzzyNeuron())
        }
        return uniqueNeurons
    }
}