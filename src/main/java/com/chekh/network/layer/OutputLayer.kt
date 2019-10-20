package com.chekh.network.layer

class OutputLayer {
    var output: Double = 0.0
        private set

    fun calculate(output: Double) {
        this.output = output
    }

    fun getError(output: Double) = this.output - output
}