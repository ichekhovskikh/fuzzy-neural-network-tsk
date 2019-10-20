package com.chekh.network.neuron

import com.chekh.network.util.Functions

class FuzzyNeuron {
    var mu: Double = 0.0
        private set
    val b: Double = 1.0
    var center: Double = 0.0
    var sigma: Double = 0.0

    fun calculateMu(x: Double): Double {
        mu = Functions.gaussian(x, b, center, sigma)
        return mu
    }
}