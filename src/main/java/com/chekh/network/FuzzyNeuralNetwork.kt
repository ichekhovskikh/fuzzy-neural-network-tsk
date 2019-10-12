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

    private fun layersRecalculate() {
        fuzzyLayer.calculate(inputLayer)
        aggregationLayer.calculate(fuzzyLayer)
        generatingLayer.calculate(inputLayer, aggregationLayer)
        summingLayer.calculate(generatingLayer, aggregationLayer)
        softmaxLayer.calculate(summingLayer)
        outputLayer.calculate(softmaxLayer)
    }

    private fun calculateLinearParams(dataset: Dataset) {
        val fullActivationMatrix = mutableListOf<DoubleArray>()
        val fullOutputMatrix = mutableListOf<Double>()
        dataset.rows.shuffled().forEach { row ->
            calcutale(row.inputs)
            val activationMatrixForOneEpoch = generatingLayer.asActivationMatrix(aggregationLayer)
            fullActivationMatrix.add(activationMatrixForOneEpoch)
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
            fuzzyLayer.correct(inputLayer, generatingLayer, error, learningRate)
        }
    }
}