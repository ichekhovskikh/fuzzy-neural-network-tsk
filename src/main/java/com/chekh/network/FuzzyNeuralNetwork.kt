package com.chekh.network

import com.chekh.network.layer.AggregationLayer
import com.chekh.network.layer.*
import com.chekh.network.util.toList
import com.chekh.network.util.toSimpleColumn
import com.chekh.network.util.toSimpleMatrix

class FuzzyNeuralNetwork(val inputCount: Int, val ruleCount: Int) {
    private val inputLayer = InputLayer(inputCount)
    private val fuzzyLayer = FuzzyLayer(inputCount, ruleCount)
    private val aggregationLayer = AggregationLayer(ruleCount)
    private val generatingLayer = GeneratingLayer(inputCount, ruleCount)
    private val summingLayer = SummingLayer()
    private val softmaxLayer = SoftmaxLayer()
    private val outputLayer = OutputLayer()

    fun retrain(dataset: Dataset, epoch: Int, learningRate: Double) {
        initWeights(dataset)
        train(dataset, epoch, learningRate)
    }

    fun train(dataset: Dataset, epoch: Int, learningRate: Double) {
        for (index in 0 until epoch) {
            calculateLinearParams(dataset)
            calculateNonLinearParams(dataset, learningRate)
        }
    }

    fun calcutale(x: List<Double>): Double {
        inputLayer.x = x
        layersRecalculate()
        return outputLayer.y
    }

    private fun initWeights(dataset: Dataset) {
        fuzzyLayer.initWeights(dataset)
        generatingLayer.initWeights()
    }

    private fun layersRecalculate() {
        fuzzyLayer.calculate(inputLayer.x)
        aggregationLayer.calculate(fuzzyLayer.muGroupedByRules)
        generatingLayer.calculate(inputLayer.x, aggregationLayer.weights)
        summingLayer.calculate(generatingLayer.generating, aggregationLayer.weights)
        softmaxLayer.calculate(summingLayer.signal, summingLayer.weightSum)
        outputLayer.calculate(softmaxLayer.y)
    }

    private fun calculateLinearParams(dataset: Dataset) {
        val fullActivationMatrix = mutableListOf<DoubleArray>()
        val fullOutputMatrix = mutableListOf<Double>()
        dataset.rows.shuffled().forEach { row ->
            calcutale(row.inputs)
            val activationLevels = aggregationLayer.asActivationArray()
            val activationArrayForOneEpoch = generatingLayer.asActivationArray(activationLevels)
            fullActivationMatrix.add(activationArrayForOneEpoch)
            fullOutputMatrix.add(row.output)
        }
        val linearParams = fullActivationMatrix
            .toSimpleMatrix()
            .pseudoInverse()
            .mult(fullOutputMatrix.toSimpleColumn())
            .toList(0)
        generatingLayer.setLinearParams(linearParams)
    }

    private fun calculateNonLinearParams(dataset: Dataset, learningRate: Double) {
        dataset.rows.shuffled().forEach { row ->
            calcutale(row.inputs)
            val error = outputLayer.getError(row.output)
            fuzzyLayer.correct(inputLayer.x, generatingLayer.pGroupedByRules, error, learningRate)
        }
    }
}