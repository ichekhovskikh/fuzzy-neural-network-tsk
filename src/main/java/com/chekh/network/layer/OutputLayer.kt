package com.chekh.network.layer

class OutputLayer {
    var y: Double = 0.0
        private set

    fun calculate(softmaxLayer: SoftmaxLayer) {
        y = softmaxLayer.neuron.y
    }

    fun getError(output: Double): Double {
        return  y - output
    }
}