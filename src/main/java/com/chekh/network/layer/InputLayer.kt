package com.chekh.network.layer

class InputLayer(val inputCount: Int) {
    var x: List<Double> = mutableListOf()
    set(value) {
        require(inputCount == value.size)
        field = value
    }
}