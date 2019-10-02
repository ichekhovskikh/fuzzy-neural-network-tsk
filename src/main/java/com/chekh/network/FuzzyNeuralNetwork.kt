package com.chekh.network

import com.chekh.network.layer.AggregationLayer
import com.chekh.network.layer.*
import com.chekh.network.util.toList
import com.chekh.network.util.toSimpleColumn
import com.chekh.network.util.toSimpleMatrix

class FuzzyNeuralNetwork(val inputCount: Int, val ruleCount: Int, val outputCount: Int) {
    private val inputLayer = InputLayer(inputCount)
    private val fuzzyLayer = FuzzyLayer(inputCount, ruleCount)
    private val aggregationLayer = AggregationLayer(ruleCount)
    private val generatingLayer = GeneratingLayer(inputCount, ruleCount, outputCount)
    private val summingLayer = SummingLayer(outputCount)
    private val softmaxLayer = SoftmaxLayer(outputCount)
    private val outputLayer = OutputLayer(outputCount)

    fun train(algorithm: LearningAlgorithm, dataset: Dataset, epoch: Int, learningRate: Double) {
        when (algorithm) {
            LearningAlgorithm.HYBRID -> hybrid(dataset, epoch, learningRate)
            LearningAlgorithm.QUICK_PROP -> quickProp(dataset, epoch, learningRate)
        }
    }

    fun calcutale(x: List<Double>): List<Double> {
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

    private fun hybrid(dataset: Dataset, epoch: Int, learningRate: Double) {
        for (index in 0 until epoch) {
            calculateHybridLinearParams(dataset)
            calculateHybridNonLinearParams(dataset, learningRate)
        }
    }

    private fun quickProp(dataset: Dataset, epoch: Int, learningRate: Double) {
        for (index in 0 until epoch) {
            dataset.rows.shuffled().forEach { row ->
                calcutale(row.inputs)
                TODO("not implemented")
            }
        }
    }

    private fun calculateHybridLinearParams(dataset: Dataset) {
        val fullActivationMatrix = mutableListOf<DoubleArray>()
        val fullOutputMatrix = mutableListOf<Double>()
        dataset.rows.shuffled().forEach { row ->
            calcutale(row.inputs)
            val activationMatrixForOneEpoch = generatingLayer.asActivationMatrix(aggregationLayer)
            fullActivationMatrix.addAll(activationMatrixForOneEpoch)
            fullOutputMatrix.addAll(row.output)
        }
        val linearParams = fullActivationMatrix
            .toSimpleMatrix()
            .pseudoInverse()
            .mult(fullOutputMatrix.toSimpleColumn())
            .toList(0)
        generatingLayer.setLinearParams(linearParams)
    }

    private fun calculateHybridNonLinearParams(dataset: Dataset, learningRate: Double) {
        dataset.rows.shuffled().forEach { row ->
            calcutale(row.inputs)
            var errors = outputLayer.getErrors(row.output)
            errors = generatingLayer.getErrors(inputLayer, errors)
            fuzzyLayer.hybridCorrect(inputLayer, errors, learningRate)
        }
    }

    enum class LearningAlgorithm {
        HYBRID,
        QUICK_PROP
    }
}