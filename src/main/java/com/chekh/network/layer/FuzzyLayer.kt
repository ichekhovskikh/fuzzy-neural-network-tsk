package com.chekh.network.layer

import com.chekh.network.neuron.FuzzyNeuron
import com.chekh.network.util.derivativeB
import com.chekh.network.util.derivativeC
import com.chekh.network.util.derivativeSigma

class FuzzyLayer(val inputCount: Int, val ruleCount: Int) {
    var neurons: MutableList<FuzzyNeuron> = mutableListOf()
        private set

    init {
        val uniqueNeurons = createUniqueNeurons(ruleCount)
        for (index in 0 until inputCount) {
            uniqueNeurons.forEach {
                neurons.add(FuzzyNeuron(it))
            }
        }
    }

    fun calculate(inputLayer: InputLayer) {
        require(inputCount == inputLayer.inputCount)
        inputLayer.x.forEachIndexed { inputIndex, value ->
            for (ruleIndex in 0 until ruleCount) {
                getFuzzyNeuron(inputIndex, ruleIndex).calculateMu(value)
            }
        }
    }

    fun correct(inputLayer: InputLayer, generatingLayer: GeneratingLayer, error: Double, learningRate: Double) {
        val muList = getMuList()
        inputLayer.x.forEachIndexed { inputIndex, value ->
            for (ruleIndex in 0 until ruleCount) {
                val fuzzyNeuron = getFuzzyNeuron(inputIndex, ruleIndex)
                var gradC = 0.0
                var gradSigma = 0.0
                var gradB = 0.0
                generatingLayer.neurons.forEachIndexed { generatingIndex, generatingNeuron ->
                    var sum = generatingNeuron.p[0]
                    inputLayer.x.forEachIndexed { index, value -> sum += generatingNeuron.p[index + 1] * value }
                    val derivativeC = fuzzyNeuron.derivativeC(ruleIndex, inputIndex, generatingIndex, value, muList)
                    val derivativeS = fuzzyNeuron.derivativeSigma(ruleIndex, inputIndex, generatingIndex, value, muList)
                    val derivativeB = fuzzyNeuron.derivativeB(ruleIndex, inputIndex, generatingIndex, value, muList)
                    gradC += error * sum * derivativeC
                    gradSigma += error * sum * derivativeS
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

    private fun getMuList(): List<List<Double>> {
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

    private fun createUniqueNeurons(count: Int): List<FuzzyNeuron> {
        val uniqueNeurons = mutableListOf<FuzzyNeuron>()
        for (index in 0 until count) {
            uniqueNeurons.add(FuzzyNeuron())
        }
        return uniqueNeurons
    }
}