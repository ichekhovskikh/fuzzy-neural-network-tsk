package com.chekh.network.layer

class OutputLayer {
    var y: Double = 0.0
        private set

    fun calculate(y: Double) {
        this.y = y
    }

    fun getError(output: Double) = y - output
}