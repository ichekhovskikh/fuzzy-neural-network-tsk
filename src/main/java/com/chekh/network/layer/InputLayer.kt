package com.chekh.network.layer

class InputLayer(val inputCount: Int) {
    var inputs: List<Double> = mutableListOf()
        set(value) {
            require(inputCount == value.size)
            field = value
        }
}