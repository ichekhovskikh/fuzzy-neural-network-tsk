package com.chekh.network

class FuzzyNeuralNetworkBuilder {
    private var inputCount = 1
    private var ruleCount = 1

    fun inputs(inputCount: Int): FuzzyNeuralNetworkBuilder = apply { this.inputCount = inputCount }

    fun rules(ruleCount: Int): FuzzyNeuralNetworkBuilder = apply { this.ruleCount = ruleCount }

    fun build(): FuzzyNeuralNetwork {
        require(inputCount > 0 && ruleCount > 0)
        return FuzzyNeuralNetwork(inputCount, ruleCount)
    }
}