package com.chekh.network.neuron

import com.chekh.network.util.Functions

class FuzzyNeuron {
    var mu: Double = 0.0
        private set
    var b: Double = 0.0
    var c: Double = 0.0
    var sigma: Double = 0.0

    fun calculateMu(x: Double): Double {
        mu = Functions.gaussian(x, b, c, sigma)
        return mu
    }
}