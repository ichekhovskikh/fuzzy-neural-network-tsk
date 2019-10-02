package com.chekh.network.layer

class OutputLayer(val outputCount: Int) {
    var y: List<Double> = mutableListOf()
        private set

    fun calculate(softmaxLayer: SoftmaxLayer) {
        require(outputCount == softmaxLayer.outputCount)
        y = softmaxLayer.neurons.map { it.y }
    }

    fun getErrors(output: List<Double>): List<Double> {
        TODO("not implemented")
    }
}