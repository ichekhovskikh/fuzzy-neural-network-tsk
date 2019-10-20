package com.chekh.network

import com.chekh.network.util.ErrorDrawer
import com.chekh.network.util.Logger

class FuzzyNeuralNetworkBuilder {
    private var inputCount = 1
    private var ruleCount = 3
    private var log: Logger? = null
    private var drawer: ErrorDrawer? = null

    fun inputs(inputCount: Int): FuzzyNeuralNetworkBuilder = apply { this.inputCount = inputCount }

    fun rules(ruleCount: Int): FuzzyNeuralNetworkBuilder = apply { this.ruleCount = ruleCount }

    fun logger(log: Logger): FuzzyNeuralNetworkBuilder = apply { this.log = log }

    fun drawer(errorDrawer: ErrorDrawer): FuzzyNeuralNetworkBuilder = apply { this.drawer = errorDrawer }

    fun build(): FuzzyNeuralNetwork {
        require(inputCount > 0 && ruleCount > 0 && ruleCount % 3 == 0)
        return FuzzyNeuralNetwork(inputCount, ruleCount).apply {
            logger = log
            errorDrawer = drawer
        }
    }
}