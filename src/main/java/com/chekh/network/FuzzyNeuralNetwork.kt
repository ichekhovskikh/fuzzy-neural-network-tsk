package com.chekh.network

import com.chekh.network.layer.AggregationLayer
import com.chekh.network.layer.*

class FuzzyNeuralNetwork(val inputCount: Int, val ruleCount: Int, val outputCount: Int) {
    private val inputLayer = InputLayer(inputCount)
    private val fuzzyLayer = FuzzyLayer(inputCount, ruleCount)
    private val aggregationLayer = AggregationLayer(ruleCount)
    private val generatingLayer = GeneratingLayer(inputCount, ruleCount, outputCount)
    private val summingLayer = SummingLayer(outputCount)
    private val softmaxLayer = SoftmaxLayer(outputCount)
    private val outputLayer = OutputLayer(outputCount)

    fun train(dataset: Dataset, epoch: Int, learningRate: Double) {

    }

    fun calcutale(x: List<Double>): List<Double> {
        inputLayer.x = x
        startLayersRecalculate()
        return outputLayer.y
    }

    private fun startLayersRecalculate() {
        fuzzyLayer.calculate(inputLayer)
        aggregationLayer.calculate(fuzzyLayer)
        generatingLayer.calculate(inputLayer, aggregationLayer)
        summingLayer.calculate(generatingLayer, aggregationLayer)
        softmaxLayer.calculate(summingLayer)
        outputLayer.calculate(softmaxLayer)
    }
}