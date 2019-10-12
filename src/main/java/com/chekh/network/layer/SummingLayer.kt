package com.chekh.network.layer

import com.chekh.network.neuron.SummingNeuron

class SummingLayer {
    private val summingNeuron = SummingNeuron()
    private val weightNeuron = SummingNeuron()
    val signal: Double get() = summingNeuron.sum
    val weightSum: Double get() = weightNeuron.sum

    fun calculate(generating: List<Double>, weights: List<Double>) {
        require(weights.size == generating.size)
        summingNeuron.calculateSum(generating)
        weightNeuron.calculateSum(weights)
    }
}