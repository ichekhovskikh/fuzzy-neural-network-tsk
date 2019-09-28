package main.java.com.chekh.network.neuron

import main.java.com.chekh.network.util.Functions.Companion.mul

class AggregationNeuron {
    var weight: Double = 0.0
        private set

    fun calculateWeight(muList: List<Double>): Double {
        weight = muList.mul()
        return weight
    }
}
