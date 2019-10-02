package com.chekh.network.neuron

import com.chekh.network.util.Functions
import java.util.concurrent.ThreadLocalRandom

class FuzzyNeuron {
    var mu: Double = 0.0
        private set
    var b: Double = 0.0
        private set
    var c: Double = 0.0
        private set
    var sigma: Double = 0.0
        private set

    constructor() {
        val random = ThreadLocalRandom.current()
        b = random.nextDouble(1.0)
        c = random.nextDouble(1.0)
        sigma = random.nextDouble(1.0)
    }

    constructor(neuron: FuzzyNeuron) {
        b = neuron.b
        c = neuron.c
        sigma = neuron.sigma
        mu = neuron.mu
    }

    fun calculateMu(x: Double): Double {
        mu = Functions.gaussian(x, b, c, sigma)
        return mu
    }
}