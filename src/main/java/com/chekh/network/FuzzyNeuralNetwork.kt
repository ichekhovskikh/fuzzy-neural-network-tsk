package com.chekh.network

import com.chekh.network.dataset.Dataset
import com.chekh.network.layer.AggregationLayer
import com.chekh.network.layer.*
import com.chekh.network.util.*

class FuzzyNeuralNetwork(override val inputCount: Int, val ruleCount: Int) : NeuralNetwork {
    private val inputLayer = InputLayer(inputCount)
    private val fuzzyLayer = FuzzyLayer(inputCount, ruleCount)
    private val aggregationLayer = AggregationLayer(ruleCount)
    private val generatingLayer = GeneratingLayer(inputCount, ruleCount)
    private val summingLayer = SummingLayer()
    private val softmaxLayer = SoftmaxLayer()
    private val outputLayer = OutputLayer()
    var logger: Logger? = null
    var errorDrawer: ErrorDrawer? = null

    override val outputCount = 1

    override fun retrain(dataset: Dataset, epoch: Int, learningRate: Double) {
        errorDrawer?.clear()
        initWeights(dataset)
        train(dataset, epoch, learningRate)
    }

    override fun train(dataset: Dataset, epoch: Int, learningRate: Double) {
        logger?.log("\nSTART TRAIN\n")
        for (index in 0 until epoch) {
            logger?.log("epoch = $index")
            correctLinearParams(dataset)
            correctNonLinearParams(dataset, learningRate)
        }
        errorDrawer?.draw()
        logger?.log("min error = ${errorDrawer?.errors?.min()} max error = ${errorDrawer?.errors?.max()}")

    }

    override fun test(dataset: Dataset, accuracyDelta: Double): Float {
        var errors = 0
        logger?.log("\nSTART TESTING")
        dataset.rows.forEach { row ->
            logger?.log("\n$row")
            val output = calculate(row.inputs)
            if (output !in row.output - accuracyDelta..row.output + accuracyDelta) errors++
            logger?.log("test: real = ${row.output} network = $output")
            logger?.log("errors = $errors accuracy = ${1f - errors.toFloat() / dataset.rows.size}")
        }
        return 1f - errors.toFloat() / dataset.rows.size
    }

    override fun calculate(inputs: List<Double>): Double {
        inputLayer.inputs = inputs
        layersRecalculate()
        return outputLayer.output
    }

    private fun initWeights(dataset: Dataset) {
        fuzzyLayer.initNonLinearParams(dataset)
        generatingLayer.initLinearParams()
    }

    private fun layersRecalculate() {
        fuzzyLayer.calculate(inputLayer.inputs)
        aggregationLayer.calculate(fuzzyLayer.muGroupedByRules)
        generatingLayer.calculate(inputLayer.inputs, aggregationLayer.weights)
        summingLayer.calculate(generatingLayer.generating, aggregationLayer.weights)
        softmaxLayer.calculate(summingLayer.signal, summingLayer.weightSum)
        outputLayer.calculate(softmaxLayer.output)
    }

    private fun correctLinearParams(dataset: Dataset) {
        val activationMatrix = mutableListOf<DoubleArray>()
        val outputMatrix = mutableListOf<Double>()
        dataset.rows.shuffled().forEach { row ->
            calculate(row.inputs)
            val activationLevels = aggregationLayer.activationLevels
            val weightedActivationLevels = generatingLayer.getWeightedActivationLevels(row.inputs, activationLevels)
            activationMatrix.add(weightedActivationLevels)
            outputMatrix.add(row.output)
        }
        val linearParams = activationMatrix
            .toSimpleMatrix()
            .pseudoInverse()
            .mult(outputMatrix.toSimpleColumn())
            .toList(0)
        generatingLayer.setLinearParams(linearParams)
    }

    private fun correctNonLinearParams(dataset: Dataset, learningRate: Double) {
        dataset.rows.shuffled().forEach { row ->
            calculate(row.inputs)
            val error = outputLayer.getError(row.output)
            errorDrawer?.errors?.add(error)
            fuzzyLayer.correct(inputLayer.inputs, generatingLayer.paramsGroupedByRules, error, learningRate)
        }
    }
}